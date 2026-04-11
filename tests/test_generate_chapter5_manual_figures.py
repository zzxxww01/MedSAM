import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_chapter5_manual_figures.py"
    )
    spec = importlib.util.spec_from_file_location("chapter5_manual_figures", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_case_arrays():
    imgs = np.stack(
        [
            np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6),
            np.linspace(1.0, 0.0, 36, dtype=np.float32).reshape(6, 6),
        ],
        axis=0,
    )

    gts = np.zeros((2, 6, 6), dtype=np.uint8)
    gts[0, 1:5, 1:5] = 4
    gts[1, 1:5, 1:5] = 4
    gts[1, 2:5, 4:6] = 12

    pred_a0 = np.zeros_like(gts)
    pred_a0[0, 2:5, 2:5] = 4
    pred_a0[1, 2:5, 2:5] = 4
    pred_a0[1, 3:5, 4:6] = 12

    pred_a3r3 = np.zeros_like(gts)
    pred_a3r3[0, 1:5, 2:5] = 4
    pred_a3r3[1, 1:5, 2:5] = 4
    pred_a3r3[1, 2:5, 4:6] = 12

    pred_c3 = np.copy(gts)

    return imgs, gts, {"A0": pred_a0, "A3R3": pred_a3r3, "C3": pred_c3}


def write_case(root: Path, case_name: str = "Case001.npz"):
    data_root = root / "data"
    pred_roots = {exp: root / exp for exp in ("A0", "A3R3", "C3")}
    data_root.mkdir(parents=True, exist_ok=True)
    for pred_root in pred_roots.values():
        pred_root.mkdir(parents=True, exist_ok=True)

    imgs, gts, preds = build_case_arrays()
    np.savez(data_root / case_name, imgs=imgs, gts=gts)
    for exp, segs in preds.items():
        np.savez(pred_roots[exp] / case_name, imgs=imgs, gts=gts, segs=segs)

    return data_root, pred_roots


class Chapter5FigureScriptTests(unittest.TestCase):
    def test_case_name_can_be_resolved_by_stem_or_substring(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root, pred_roots = write_case(root, case_name="FLARE22_Tr_0001.npz")

            index = module.build_case_index(
                str(data_root),
                {name: str(path) for name, path in pred_roots.items()},
            )

            self.assertEqual(
                module.resolve_case_name(index, "FLARE22_Tr_0001"),
                "FLARE22_Tr_0001.npz",
            )
            self.assertEqual(
                module.resolve_case_name(index, "0001"),
                "FLARE22_Tr_0001.npz",
            )

    def test_can_list_candidates_and_build_case_index(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root, pred_roots = write_case(root)

            index = module.build_case_index(
                str(data_root),
                {name: str(path) for name, path in pred_roots.items()},
            )

            candidates = module.list_failure_candidates(
                index,
                organ_ids=[4],
                dice_lower=0.0,
                dice_upper=1.0,
                min_area=1,
                limit=10,
                reference_experiment="A0",
            )

            self.assertEqual(index.case_names, ["Case001.npz"])
            self.assertTrue(candidates)
            self.assertEqual(candidates[0]["case_name"], "Case001.npz")
            self.assertEqual(candidates[0]["organ_id"], 4)
            self.assertEqual(module.list_cases(index, contains="Case", limit=5), ["Case001.npz"])

    def test_can_render_mask_boundary_and_failure_figures(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root, pred_roots = write_case(root)
            output_dir = root / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            index = module.build_case_index(
                str(data_root),
                {name: str(path) for name, path in pred_roots.items()},
            )

            module.render_mask_comparison(
                index,
                {
                    "case_name": "Case001.npz",
                    "slice_idx": 1,
                    "output_path": str(output_dir / "mask.pdf"),
                    "columns": ["GT", "A0", "A3R3", "C3"],
                    "titles": {
                        "GT": "GT",
                        "A0": "Baseline",
                        "A3R3": "BL",
                        "C3": "BL+MSL",
                    },
                },
            )
            module.render_boundary_detail(
                index,
                {
                    "case_name": "Case001.npz",
                    "slice_idx": 1,
                    "output_path": str(output_dir / "boundary.pdf"),
                    "columns": ["GT", "A0", "A3R3", "C3"],
                    "roi": [1, 5, 1, 5],
                    "titles": {
                        "GT": "GT",
                        "A0": "Baseline",
                        "A3R3": "BL",
                        "C3": "BL+MSL",
                    },
                },
            )
            module.render_failure_cases(
                index,
                {
                    "output_path": str(output_dir / "failure.pdf"),
                    "rows": [
                        {
                            "case_name": "Case001.npz",
                            "slice_idx": 0,
                            "organ_id": 4,
                            "row_label": "(a) Weak boundary\n(Pancreas)",
                        },
                        {
                            "case_name": "Case001.npz",
                            "slice_idx": 1,
                            "organ_id": 4,
                            "row_label": "(b) Small organ\n(Pancreas)",
                        },
                        {
                            "case_name": "Case001.npz",
                            "slice_idx": 1,
                            "organ_id": 12,
                            "row_label": "(c) Adjacent organs\n(Duodenum)",
                        },
                    ],
                    "columns": ["GT", "A0", "A3R3", "C3"],
                    "titles": {
                        "GT": "GT",
                        "A0": "Baseline",
                        "A3R3": "BL",
                        "C3": "BL+MSL",
                    },
                },
            )

            self.assertTrue((output_dir / "mask.pdf").exists())
            self.assertTrue((output_dir / "boundary.pdf").exists())
            self.assertTrue((output_dir / "failure.pdf").exists())


if __name__ == "__main__":
    unittest.main()
