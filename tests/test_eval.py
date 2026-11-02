"""Controlled errors test intended metric behavior, not just code execution."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval.io import Record, load_records
from eval.matching import assign
from eval.metrics import aggregate, evaluate_sample
from eval.schema import parse_structure
from eval.score import evaluate_dataset, main


def fixture():
    return {
        "coordinate_mode": "normalized_1000",
        "background": {"type": "photographic", "id": "bg"},
        "objects": [
            {"id": "p", "kind": "photo", "bbox": [100, 100, 400, 400]},
            {"id": "f", "kind": "frame", "bbox": [80, 80, 420, 420], "photo_id": ["p"]},
            {"id": "cut", "kind": "photo", "bbox": [200, 50, 450, 350], "linked_to": "p"},
            {"id": "star", "kind": "overlay", "bbox": [600, 600, 700, 700]},
            {"id": "txt", "kind": "text", "bbox": [100, 800, 900, 900]},
        ],
        "layer_order": ["p", "f", "cut", "star", "txt"],
    }


PAIRS = [["p", "f"], ["f", "cut"]]


def score(pred, gt=None, pairs=PAIRS, **parse_kwargs):
    return evaluate_sample("sample", parse_structure(gt or fixture()),
                           parse_structure(pred, **parse_kwargs), pairs)


class MetricBehaviorTests(unittest.TestCase):
    def test_perfect_and_renamed_ids(self):
        g = fixture()
        p = deepcopy(g)
        names = {x: "new_" + x for x in ["p", "f", "cut", "star", "txt", "bg"]}
        p["background"]["id"] = names["bg"]
        for o in p["objects"]:
            o["id"] = names[o["id"]]
            if "photo_id" in o:
                o["photo_id"] = [names[x] for x in o["photo_id"]]
            if "linked_to" in o:
                o["linked_to"] = names[o["linked_to"]]
        p["objects"].reverse()
        p["layer_order"] = [names[x] for x in g["layer_order"]]
        s = score(p)
        self.assertEqual(s["detection"]["f1"], 1)
        self.assertEqual(s["classification"]["accuracy"], 1)
        self.assertEqual(s["localization"]["mean_iou"], 1)
        self.assertEqual(s["relations"]["photo_id"]["f1"], 1)
        self.assertEqual(s["relations"]["linked_to"]["f1"], 1)
        self.assertEqual(s["occlusion_order"]["accuracy"], 1)
        self.assertTrue(s["validity"]["structure_valid"])

    def test_classification_is_separate_from_detection(self):
        p = fixture()
        p["objects"][3]["kind"] = "text"
        s = score(p)
        self.assertEqual(s["detection"]["f1"], 1)
        self.assertEqual(s["classification"]["accuracy"], 0.8)
        self.assertEqual(s["localization"]["mean_iou"], 1)
        self.assertEqual(s["classification"]["confusion"]["overlay"]["text"], 1)
        self.assertEqual(s["auxiliary_iou_f1"]["overlay"]["iou_weighted_f1"], 0)

    def test_wrong_endpoint_class_fails_relations_and_order(self):
        p = fixture()
        p["objects"][0]["kind"] = "overlay"
        s = score(p)
        self.assertEqual(s["detection"]["f1"], 1)
        self.assertEqual(s["relations"]["photo_id"]["f1"], 0)
        self.assertEqual(s["relations"]["linked_to"]["f1"], 0)
        self.assertEqual(s["occlusion_order"]["accuracy"], 0.5)

    def test_missed_element_lowers_detection_and_order(self):
        p = fixture()
        p["objects"].pop(0)
        p["layer_order"].remove("p")
        s = score(p)
        self.assertAlmostEqual(s["detection"]["f1"], 8 / 9)
        self.assertEqual(s["localization"]["coverage"], 0.8)
        self.assertEqual(s["localization"]["mean_iou"], 1)
        self.assertEqual(s["occlusion_order"]["accuracy"], 0.5)
        self.assertEqual(s["occlusion_order"]["conditional_accuracy"], 1)

    def test_duplicate_prediction_is_false_positive(self):
        p = fixture()
        duplicate = dict(p["objects"][0], id="duplicate")
        p["objects"].append(duplicate)
        p["layer_order"].append("duplicate")
        s = score(p)
        self.assertEqual(s["detection"]["tp"], 5)
        self.assertEqual(s["detection"]["fp"], 1)
        self.assertAlmostEqual(s["detection"]["f1"], 10 / 11)
        self.assertAlmostEqual(s["auxiliary_iou_f1"]["photo"]["iou_weighted_f1"], 0.8)
        self.assertTrue(s["details"]["potential_geometry_ties"])

    def test_wrong_relation_counts_fp_and_fn(self):
        p = fixture()
        p["objects"][1]["photo_id"] = ["cut"]
        s = score(p)
        self.assertEqual(s["relations"]["photo_id"]["fp"], 1)
        self.assertEqual(s["relations"]["photo_id"]["fn"], 1)
        self.assertEqual(s["relations"]["linked_to"]["f1"], 1)

    def test_duplicate_relation_only_gets_one_credit(self):
        p = fixture()
        p["objects"][1]["photo_id"] = ["p", "p"]
        s = score(p)
        self.assertEqual(s["relations"]["photo_id"]["tp"], 1)
        self.assertEqual(s["relations"]["photo_id"]["fp"], 1)
        self.assertAlmostEqual(s["relations"]["photo_id"]["f1"], 2 / 3)
        self.assertFalse(s["validity"]["structure_valid"])

    def test_invalid_reference_is_false_positive(self):
        p = fixture()
        p["objects"][2]["linked_to"] = "does_not_exist"
        s = score(p)
        self.assertEqual(s["relations"]["linked_to"]["fp"], 1)
        self.assertEqual(s["relations"]["linked_to"]["fn"], 1)
        self.assertFalse(s["validity"]["structure_valid"])

    def test_photographic_background_reference_and_wrong_background(self):
        g = fixture()
        g["objects"][2]["linked_to"] = "bg"
        p = deepcopy(g)
        p["background"]["id"] = "renamed_bg"
        p["objects"][2]["linked_to"] = "renamed_bg"
        self.assertEqual(score(p, g)["relations"]["linked_to"]["f1"], 1)
        p["background"]["type"] = "reference_clean"
        s = score(p, g)
        self.assertEqual(s["relations"]["linked_to"]["f1"], 0)
        self.assertFalse(s["background"]["correct"])

    def test_reversed_order(self):
        p = fixture()
        p["layer_order"] = ["p", "cut", "f", "star", "txt"]
        s = score(p)
        self.assertEqual(s["detection"]["f1"], 1)
        self.assertEqual(s["occlusion_order"]["accuracy"], 0.5)

    def test_missing_or_duplicate_order_id(self):
        for replacement in (["p", "cut", "star", "txt"],
                            ["p", "f", "f", "cut", "star", "txt"]):
            p = fixture()
            p["layer_order"] = replacement
            s = score(p)
            self.assertEqual(s["occlusion_order"]["accuracy"], 0)
            self.assertFalse(s["validity"]["structure_valid"])

    def test_unknown_class_is_counted_as_wrong(self):
        p = fixture()
        p["objects"][3]["kind"] = "sticker"
        s = score(p)
        self.assertEqual(s["detection"]["f1"], 1)
        self.assertEqual(s["classification"]["accuracy"], 0.8)
        self.assertEqual(s["classification"]["confusion"]["overlay"]["__invalid__"], 1)

    def test_invalid_box_is_not_dropped_or_clipped(self):
        p = fixture()
        p["objects"][3]["bbox"] = [-10, 600, 700, 700]
        s = score(p)
        self.assertEqual(s["detection"]["pred_count"], 5)
        self.assertEqual(s["detection"]["fp"], 1)
        self.assertEqual(s["detection"]["fn"], 1)
        self.assertFalse(s["validity"]["structure_valid"])

    def test_malformed_or_missing_output_is_kept(self):
        s = score("not json")
        self.assertEqual(s["detection"]["recall"], 0)
        self.assertEqual(s["detection"]["f1"], 0)
        self.assertIsNone(s["classification"]["accuracy"])
        self.assertIsNone(s["localization"]["mean_iou"])
        self.assertEqual(s["occlusion_order"]["accuracy"], 0)
        self.assertFalse(s["validity"]["json_valid"])
        summary, _ = evaluate_dataset({"a": Record("a", fixture())}, {})
        self.assertEqual(summary["num_examples"], 1)
        self.assertEqual(summary["validity"]["present_rate"], 0)
        self.assertEqual(summary["background"]["accuracy"], 0)

    def test_unknown_order_gt_is_not_scored_as_perfect(self):
        s = score(fixture(), pairs=None)
        self.assertIsNone(s["occlusion_order"]["accuracy"])
        self.assertFalse(s["occlusion_order"]["reviewed"])
        empty = score(fixture(), pairs=[])
        self.assertIsNone(empty["occlusion_order"]["accuracy"])
        self.assertTrue(empty["occlusion_order"]["reviewed"])
        self.assertEqual(aggregate([s, empty])["occlusion_order"]["unreviewed_images"], 1)

    def test_invalid_gt_and_contradictory_pairs_fail(self):
        g = fixture()
        g["objects"][0]["bbox"] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "Invalid GT"):
            score(fixture(), g)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            score(fixture(), pairs=[["f", "p"]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score(fixture(), pairs=[["p", "f"], ["p", "f"]])

    def test_pixel_and_normalized_1_coordinates(self):
        p = fixture()
        p["coordinate_mode"] = "pixel"
        for o in p["objects"]:
            o["bbox"] = [v * (2 if j % 2 == 0 else 1) for j, v in enumerate(o["bbox"])]
        self.assertEqual(score(p, width=2000, height=1000)["localization"]["mean_iou"], 1)
        self.assertEqual(score(p)["detection"]["f1"], 0)
        p = fixture()
        p["coordinate_mode"] = "normalized_1"
        for o in p["objects"]:
            o["bbox"] = [v / 1000 for v in o["bbox"]]
        self.assertAlmostEqual(score(p)["localization"]["mean_iou"], 1)

    def test_soft_iou_has_no_point_five_cutoff(self):
        g = {"background": {"type": "transparent"},
             "objects": [{"id": "p", "kind": "photo", "bbox": [0, 0, 100, 100]}],
             "layer_order": ["p"]}
        p = deepcopy(g)
        p["objects"][0]["bbox"] = [0, 0, 49, 100]
        s = score(p, g, pairs=[])
        self.assertEqual(s["detection"]["f1"], 0)
        self.assertAlmostEqual(s["auxiliary_iou_f1"]["photo"]["iou_weighted_f1"], 0.49)
        self.assertIsNone(s["auxiliary_iou_f1"]["frame"]["iou_weighted_f1"])

    def test_hard_matching_prioritizes_cardinality(self):
        iou = np.array([[1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 0]])
        self.assertEqual(len(assign(iou, threshold=0.5)), 3)
        self.assertEqual(len(assign(iou)), 2)

    def test_template_mean_does_not_let_dense_images_dominate(self):
        large = {"background": {"type": "transparent"},
                 "objects": [{"id": f"p{i}", "kind": "photo",
                              "bbox": [i * 40, 0, i * 40 + 20, 20]} for i in range(20)],
                 "layer_order": [f"p{i}" for i in range(20)]}
        small = deepcopy(large)
        small["objects"] = small["objects"][:2]
        small["layer_order"] = small["layer_order"][:2]
        wrong = deepcopy(small)
        for o in wrong["objects"]:
            o["bbox"] = [0, 900, 20, 920]
        summary = aggregate([score(large, large, pairs=[]), score(wrong, small, pairs=[])])
        self.assertAlmostEqual(summary["detection"]["f1"], 20 / 22)
        self.assertEqual(summary["per_template_macro"]["detection"]["f1"]["mean"], 0.5)
        self.assertEqual(summary["per_template_macro"]["classification"]["accuracy"]["eligible_images"], 1)

    def test_macro_relation_empty_and_hallucinated_cases(self):
        empty = {"background": {"type": "transparent"}, "objects": [], "layer_order": []}
        hallucinated = deepcopy(empty)
        hallucinated["objects"] = [
            {"id": "p", "kind": "photo", "bbox": [0, 0, 20, 20], "linked_to": "missing"}]
        hallucinated["layer_order"] = ["p"]
        summary = aggregate([score(empty, empty, pairs=[]),
                             score(hallucinated, empty, pairs=[])])
        value = summary["per_template_macro"]["relations"]["linked_to"]["f1"]
        self.assertEqual(value, {"mean": 0.0, "eligible_images": 1, "total_images": 2})
        self.assertIsNone(summary["per_template_macro"]["occlusion_order"]["accuracy"]["mean"])

    def test_no_free_credit_for_empty_relations_or_classes(self):
        g = {"background": {"type": "transparent"}, "objects": [], "layer_order": []}
        s = score(g, g, pairs=[])
        self.assertIsNone(s["relations"]["linked_to"]["f1"])
        self.assertIsNone(s["detection"]["f1"])
        self.assertIsNone(s["auxiliary_iou_f1"]["text"]["iou_weighted_f1"])
        p = deepcopy(g)
        p["objects"] = [{"id": "p", "kind": "photo", "bbox": [0, 0, 100, 100], "linked_to": "fake"}]
        p["layer_order"] = ["p"]
        self.assertEqual(score(p, g, pairs=[])["relations"]["linked_to"]["f1"], 0)


class InputAndReportTests(unittest.TestCase):
    def test_jsonl_and_cli_report_with_one_missing_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gt = root / "gt.jsonl"
            pred = root / "pred.jsonl"
            gt.write_text("\n".join(json.dumps({"id": key, "structure": fixture(),
                                                "occlusion_pairs": PAIRS}) for key in ("a", "b")), encoding="utf-8")
            pred.write_text(json.dumps({"id": "a", "structure": fixture()}), encoding="utf-8")
            self.assertEqual(main(["--gt", str(gt), "--pred", str(pred), "--out", str(root / "out")]), 0)
            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 2)
            self.assertEqual(summary["detection"]["recall"], 0.5)
            self.assertAlmostEqual(summary["detection"]["f1"], 2 / 3)
            self.assertEqual(summary["occlusion_order"]["accuracy"], 0.5)
            self.assertEqual(summary["per_template_macro"]["detection"]["f1"]["mean"], 0.5)
            self.assertEqual(summary["per_template_macro"]["classification"]["accuracy"]["eligible_images"], 1)
            self.assertEqual(len((root / "out" / "per_sample.jsonl").read_text(encoding="utf-8").splitlines()), 2)
            self.assertIn("50.00%", (root / "out" / "report.md").read_text(encoding="utf-8"))

    def test_duplicate_and_unexpected_ids_fail(self):
        gt = {"a": Record("a", fixture())}
        with self.assertRaisesRegex(ValueError, "unexpected"):
            evaluate_dataset(gt, {"other": Record("other", fixture())})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pred.jsonl"
            row = json.dumps({"id": "a", "structure": fixture()})
            p.write_text(row + "\n" + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_records(p)

    def test_require_occlusions(self):
        gt = {"a": Record("a", fixture())}
        with self.assertRaisesRegex(ValueError, "missing reviewed"):
            evaluate_dataset(gt, gt, require_occlusions=True)
        s, _ = evaluate_dataset(gt, gt, occlusions={"a": []}, require_occlusions=True)
        self.assertEqual(s["occlusion_order"]["reviewed_images"], 1)

    def test_manifest_ids_map_to_case_directory_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case = root / "cases" / "0001_db-curated"
            case.mkdir(parents=True)
            (case / "structure.json").write_text(json.dumps(fixture()), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"total": 1, "cases": [
                {"test_id": "0001", "structure.json": {"path": "cases/0001_db-curated/structure.json"}}
            ]}), encoding="utf-8")
            gt, _ = load_records(root)
            gt_from_file, _ = load_records(root / "manifest.json")
            self.assertEqual(gt, gt_from_file)
            pred, _ = load_records(root / "cases")
            summary, _ = evaluate_dataset(gt, pred)
            self.assertEqual(summary["detection"]["f1"], 1)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["cases"][0]["structure.json"]["sha256"] = "wrong_hash"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_records(root)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_records(root / "manifest.json")

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        for text in ('{"objects":[],"objects":[]}', '{"objects":NaN}'):
            s = parse_structure(text)
            self.assertFalse(s.json_valid)


if __name__ == "__main__":
    unittest.main()
