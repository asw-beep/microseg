"""U-Net, losses and inference tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from microseg.config import LossConfig, ModelConfig
from microseg.data.synthetic import synthetic_field
from microseg.inference import predict_probabilities
from microseg.models import CombinedLoss, SoftDiceLoss, UNet, build_loss, build_model
from microseg.models.unet import count_parameters


class TestUNet:
    def test_output_matches_input_resolution(self):
        """Padded convolutions: the segmentation map must align with the image."""
        model = UNet(1, 3, base_filters=8, depth=3)
        out = model(torch.randn(2, 1, 64, 64))
        assert out.shape == (2, 3, 64, 64)

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_depth_is_configurable(self, depth):
        model = UNet(1, 3, base_filters=4, depth=depth)
        size = 2**depth * 4
        assert model(torch.randn(1, 1, size, size)).shape == (1, 3, size, size)

    def test_multi_channel_input_is_supported(self):
        """A multiplexed assay feeds several stains as a stack."""
        model = UNet(in_channels=4, out_channels=3, base_filters=4, depth=2)
        assert model(torch.randn(1, 4, 32, 32)).shape == (1, 3, 32, 32)

    def test_non_square_input_is_handled(self):
        model = UNet(1, 3, base_filters=4, depth=2)
        assert model(torch.randn(1, 1, 32, 48)).shape == (1, 3, 32, 48)

    def test_size_divisor_reflects_depth(self):
        assert UNet(depth=4, base_filters=4).size_divisor == 16
        assert UNet(depth=2, base_filters=4).size_divisor == 4

    @pytest.mark.parametrize("norm", ["batch", "instance", "none"])
    def test_norm_variants_run(self, norm):
        model = UNet(1, 3, base_filters=4, depth=2, norm=norm)
        assert model(torch.randn(2, 1, 32, 32)).shape == (2, 3, 32, 32)

    def test_bilinear_upsampling_runs(self):
        model = UNet(1, 3, base_filters=4, depth=2, bilinear_upsample=True)
        assert model(torch.randn(1, 1, 32, 32)).shape == (1, 3, 32, 32)

    def test_gradients_reach_the_first_layer(self):
        """Guards against a detached skip connection silently killing training."""
        model = UNet(1, 3, base_filters=4, depth=2)
        loss = model(torch.randn(1, 1, 32, 32)).sum()
        loss.backward()
        first = model.stem.block[0].weight
        assert first.grad is not None and torch.isfinite(first.grad).all()
        assert first.grad.abs().sum() > 0

    def test_wider_model_has_more_parameters(self):
        assert count_parameters(UNet(base_filters=32, depth=4)) > count_parameters(
            UNet(base_filters=16, depth=4)
        )

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be"):
            UNet(depth=0)

    def test_builder_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="unknown model"):
            build_model(ModelConfig(name="resnet"))


class TestLosses:
    def test_dice_is_zero_for_a_perfect_prediction(self):
        target = torch.randint(0, 3, (2, 16, 16))
        # Very confident logits at the correct class.
        logits = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
        assert SoftDiceLoss()(logits, target).item() < 0.01

    def test_dice_is_high_for_a_wrong_prediction(self):
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        logits = torch.zeros(1, 3, 16, 16)
        logits[:, 1] = 50  # confidently predicts the wrong class everywhere
        assert SoftDiceLoss()(logits, target).item() > 0.5

    def test_combined_loss_is_positive_and_differentiable(self):
        logits = torch.randn(2, 3, 16, 16, requires_grad=True)
        loss = CombinedLoss(1.0, 1.0, [1.0, 1.0, 3.0])(logits, torch.randint(0, 3, (2, 16, 16)))
        loss.backward()
        assert loss.item() > 0 and logits.grad is not None

    def test_class_weights_penalise_the_weighted_class_more(self):
        """The boundary class is upweighted; getting it wrong must cost more."""
        target = torch.full((1, 8, 8), 2, dtype=torch.long)  # all boundary
        logits = torch.zeros(1, 3, 8, 8)
        logits[:, 0] = 5.0  # confidently wrong

        weighted = CombinedLoss(1.0, 0.0, [1.0, 1.0, 3.0])(logits, target)
        unweighted = CombinedLoss(1.0, 0.0, [1.0, 1.0, 1.0])(logits, target)
        assert weighted.item() > unweighted.item()

    def test_weights_move_with_the_module(self):
        loss = build_loss(LossConfig()).to("cpu")
        assert loss.class_weights.device.type == "cpu"

    def test_mismatched_class_weights_raise(self):
        with pytest.raises(ValueError, match="class_weights"):
            build_loss(LossConfig(class_weights=[1.0, 1.0]), n_classes=3)


@pytest.fixture(scope="module")
def briefly_trained_model():
    """A tiny U-Net taken just far enough to produce a structured prediction.

    Some properties -- that test-time augmentation *refines* a segmentation
    rather than replacing it -- are only properties of a model that has learned
    something. On random weights the output is noise, and any agreement measured
    against it is luck. Forty steps on one synthetic field is enough for the
    argmax map to be driven by the image instead of the initialisation, and
    costs about a second, once for the whole module.
    """
    from microseg.data.dataset import instances_to_semantic

    torch.manual_seed(0)
    image, labels = synthetic_field(shape=(64, 64), n_objects=6, seed=0)
    inputs = torch.from_numpy(image[None, None].astype(np.float32))
    target = torch.from_numpy(instances_to_semantic(labels, 2)[None]).long()

    model = build_model(ModelConfig(base_filters=4, depth=2))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(40):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), target).backward()
        optimiser.step()
    return model.eval()


class TestInference:
    def test_probabilities_sum_to_one_and_keep_shape(self):
        model = build_model(ModelConfig(base_filters=4, depth=2))
        probs = predict_probabilities(model, np.random.rand(48, 48).astype(np.float32))
        assert probs.shape == (3, 48, 48)
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)

    def test_non_divisible_input_is_padded_and_cropped_back(self):
        """A depth-4 U-Net needs multiples of 16; 100x70 is neither."""
        model = build_model(ModelConfig(base_filters=4, depth=4))
        probs = predict_probabilities(model, np.random.rand(100, 70).astype(np.float32))
        assert probs.shape == (3, 100, 70)

    def test_tta_preserves_shape_and_normalisation(self):
        model = build_model(ModelConfig(base_filters=4, depth=2))
        image = np.random.rand(32, 32).astype(np.float32)
        probs = predict_probabilities(model, image, tta=True)
        assert probs.shape == (3, 32, 32)
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)

    def test_tta_output_is_exactly_equivariant(self):
        """The property that proves the inverse transforms are applied correctly.

        Averaging over a closed symmetry group makes the *whole* prediction
        equivariant: rotating the input and rotating the TTA output back must
        give the same answer, exactly.  A missing or wrong inverse transform
        breaks this, whereas a plain forward pass satisfies it only by accident.
        """
        model = build_model(ModelConfig(base_filters=4, depth=2))
        model.eval()
        image = np.random.default_rng(0).random((32, 32)).astype(np.float32)

        direct = predict_probabilities(model, image, tta=True)
        rotated = predict_probabilities(model, np.rot90(image).copy(), tta=True)
        # Undo the rotation on the spatial axes of the (C, H, W) output.
        assert np.allclose(np.rot90(rotated, -1, axes=(1, 2)), direct, atol=1e-5)

    def test_tta_and_plain_inference_agree_on_the_argmax_class(self, briefly_trained_model):
        """TTA refines a prediction; it must not produce a different segmentation.

        Both halves of the setup here used to be wrong, and the test failed in
        CI about one run in ten. The model was untrained, so its argmax map was
        noise and "agreement" was decided by an unseeded initialisation; and the
        input was uniform random pixels, so nothing in it was in distribution.
        Measured across random inits that gave 0.43 to 0.99 against a 0.5
        threshold -- a coin flip dressed as an assertion. A briefly trained
        model on an actual field stays above 0.92.
        """
        image = synthetic_field(shape=(64, 64), n_objects=5, seed=11)[0].astype(np.float32)
        plain = predict_probabilities(briefly_trained_model, image, tta=False).argmax(0)
        augmented = predict_probabilities(briefly_trained_model, image, tta=True).argmax(0)
        assert (plain == augmented).mean() > 0.85


class TestCheckpointRoundTrip:
    def test_checkpoint_rebuilds_an_identical_model(self, tmp_path):
        """A checkpoint carries its own config, so it is self-describing."""
        from microseg.config import ExperimentConfig
        from microseg.inference import load_checkpoint
        from microseg.train import save_checkpoint

        cfg = ExperimentConfig(model=ModelConfig(base_filters=4, depth=2))
        model = build_model(cfg.model)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, cfg, epoch=0, metrics={"val_ap": 0.5})

        restored, restored_cfg, payload = load_checkpoint(path)
        assert restored_cfg.model.base_filters == 4
        assert payload["metrics"]["val_ap"] == 0.5

        image = np.random.rand(32, 32).astype(np.float32)
        assert np.allclose(
            predict_probabilities(model, image), predict_probabilities(restored, image), atol=1e-6
        )


class TestDistanceHead:
    """The optional distance head: an extra channel with a different activation.

    Every bug this class guards against is silent -- a softmax over the wrong
    slice, a distance loss that never reaches its own conv, a checkpoint that
    rebuilds the model without the head -- so the assertions are about channel
    counts, gradients and value ranges rather than accuracy.
    """

    def test_head_adds_exactly_one_channel(self):
        plain = build_model(ModelConfig(base_filters=4, depth=2))
        with_head = build_model(ModelConfig(base_filters=4, depth=2, distance_head=True))
        x = torch.randn(2, 1, 32, 32)
        assert plain(x).shape == (2, 3, 32, 32)
        assert with_head(x).shape == (2, 4, 32, 32)
        assert with_head.n_classes == 3 and with_head.has_distance
        assert not plain.has_distance

    def test_split_output_separates_the_heads(self):
        from microseg.models.losses import split_output

        logits = torch.randn(2, 4, 8, 8)
        class_logits, distance_logits = split_output(logits, 3)
        assert class_logits.shape == (2, 3, 8, 8)
        assert distance_logits.shape == (2, 1, 8, 8)
        assert torch.equal(distance_logits[:, 0], logits[:, 3])

    def test_split_output_returns_none_without_a_head(self):
        from microseg.models.losses import split_output

        class_logits, distance_logits = split_output(torch.randn(2, 3, 8, 8), 3)
        assert distance_logits is None
        assert class_logits.shape == (2, 3, 8, 8)

    def test_split_output_rejects_an_unexpected_width(self):
        from microseg.models.losses import split_output

        with pytest.raises(ValueError, match="channels"):
            split_output(torch.randn(2, 6, 8, 8), 3)

    def test_masked_l1_ignores_pixels_outside_the_mask(self):
        from microseg.models.losses import masked_l1

        prediction = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[0, :2] = True  # half the pixels are foreground
        # Error is 1 everywhere, but only the masked half is averaged over.
        assert torch.isclose(masked_l1(prediction, target, mask), torch.tensor(1.0))

        prediction[0, 0, :2] = 1.0  # make the masked half perfect
        assert torch.isclose(masked_l1(prediction, target, mask), torch.tensor(0.0))

    def test_masked_l1_is_zero_for_an_empty_mask(self):
        """An all-background tile is real data, and must not produce NaN."""
        from microseg.models.losses import masked_l1

        loss = masked_l1(
            torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.bool)
        )
        assert float(loss) == 0.0

    def test_build_loss_ignores_distance_weight_without_a_head(self):
        cfg = LossConfig(distance_weight=1.0)
        assert build_loss(cfg, 3, distance_head=False).distance_weight == 0.0
        assert build_loss(cfg, 3, distance_head=True).distance_weight == 1.0

    def test_distance_term_reaches_the_distance_conv(self):
        """The gradient check that proves the head is trained, not just present."""
        model = build_model(ModelConfig(base_filters=4, depth=2, distance_head=True))
        criterion = build_loss(
            LossConfig(ce_weight=0.0, dice_weight=0.0, distance_weight=1.0, class_weights=None),
            3,
            distance_head=True,
        )
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[:, 4:12, 4:12] = 1
        distance = torch.zeros(1, 1, 16, 16)
        distance[:, :, 6:10, 6:10] = 1.0

        loss = criterion(model(torch.randn(1, 1, 16, 16)), target, distance)
        loss.backward()
        assert model.distance.weight.grad is not None
        assert float(model.distance.weight.grad.abs().sum()) > 0

    def test_distance_loss_is_lower_for_a_better_prediction(self):
        from microseg.models.losses import CombinedLoss

        criterion = CombinedLoss(ce_weight=0.0, dice_weight=0.0, distance_weight=1.0)
        target = torch.ones(1, 8, 8, dtype=torch.long)
        distance = torch.full((1, 1, 8, 8), 0.9)
        # Channel 3 is the distance logit; sigmoid(3) ~ 0.95, sigmoid(-3) ~ 0.05.
        good = torch.zeros(1, 4, 8, 8)
        good[:, 3] = 3.0
        bad = torch.zeros(1, 4, 8, 8)
        bad[:, 3] = -3.0
        assert float(criterion(good, target, distance)) < float(criterion(bad, target, distance))

    def test_predict_maps_returns_a_sigmoid_distance_channel(self):
        from microseg.inference import predict_maps

        model = build_model(ModelConfig(base_filters=4, depth=2, distance_head=True))
        prediction = predict_maps(model, np.random.rand(48, 40).astype(np.float32))
        assert prediction.probs.shape == (3, 48, 40)
        assert np.allclose(prediction.probs.sum(axis=0), 1.0, atol=1e-5)
        assert prediction.distance is not None
        assert prediction.distance.shape == (48, 40)
        assert prediction.distance.min() >= 0.0 and prediction.distance.max() <= 1.0

    def test_predict_maps_returns_no_distance_without_a_head(self):
        from microseg.inference import predict_maps

        model = build_model(ModelConfig(base_filters=4, depth=2))
        assert predict_maps(model, np.random.rand(32, 32).astype(np.float32)).distance is None

    def test_tta_keeps_the_distance_channel_equivariant(self):
        """A scalar field rotates with the image; the class maps are already covered."""
        from microseg.inference import predict_maps

        model = build_model(ModelConfig(base_filters=4, depth=2, distance_head=True))
        model.eval()
        image = np.random.default_rng(2).random((32, 32)).astype(np.float32)

        direct = predict_maps(model, image, tta=True).distance
        rotated = predict_maps(model, np.rot90(image).copy(), tta=True).distance
        assert np.allclose(np.rot90(rotated, -1), direct, atol=1e-5)

    def test_checkpoint_round_trip_keeps_the_head(self, tmp_path):
        from microseg.config import ExperimentConfig
        from microseg.inference import load_checkpoint, predict_maps
        from microseg.train import save_checkpoint

        cfg = ExperimentConfig(model=ModelConfig(base_filters=4, depth=2, distance_head=True))
        model = build_model(cfg.model)
        path = tmp_path / "distance.pt"
        save_checkpoint(path, model, cfg, epoch=0, metrics={"val_ap": 0.1})

        restored, restored_cfg, _ = load_checkpoint(path)
        assert restored_cfg.model.distance_head is True
        assert restored.has_distance

        image = np.random.rand(32, 32).astype(np.float32)
        assert np.allclose(
            predict_maps(model, image).distance,
            predict_maps(restored, image).distance,
            atol=1e-6,
        )
