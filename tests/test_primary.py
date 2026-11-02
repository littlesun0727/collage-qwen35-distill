"""Regression tests for localization, grouping and first-pass provenance."""
import unittest
import tempfile
import json
from pathlib import Path
from copy import deepcopy
from eval.io import Record
from eval.score import evaluate_dataset
from scripts.evaluate_batch import final_prediction


def structure(kind="overlay", box=None):
    return {"background": {"type": "transparent"},
            "objects": [{"id": "a", "kind": kind, "bbox": box or [0, 0, 100, 100]}],
            "layer_order": ["a"]}


class PrimaryTests(unittest.TestCase):
    def run_score(self, gt, pred, stage="raw_first"):
        return evaluate_dataset({"x": Record("x", gt)},
                                {} if pred is None else {"x": Record("x", pred)},
                                prediction_stage=stage)[0]

    def test_found_but_loose_box(self):
        s = self.run_score(structure(), structure(box=[0, 0, 150, 100]))["primary"]
        self.assertEqual(s["recognition"]["fixed"]["mean"], 1)
        self.assertAlmostEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], 2/3)
        self.assertEqual(s["localization"]["fixed"]["f1_75"]["mean"], 0)

    def test_below_threshold_still_gets_soft_credit(self):
        s = self.run_score(structure(), structure(box=[0, 0, 400, 100]))["primary"]
        self.assertEqual(s["recognition"]["fixed"]["mean"], 0)
        self.assertEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], .25)

    def test_text_overlay_detection_and_treatment_separate(self):
        s = self.run_score(structure(), structure("text"))["primary"]
        self.assertEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], 1)
        self.assertEqual(s["treatment"]["kind_agreement_proxy"]["mean"], 0)

    def test_photo_cannot_be_matched_to_fixed(self):
        s = self.run_score(structure("photo"), structure())["primary"]
        for g in ("photo", "fixed"):
            self.assertEqual(s["localization"][g]["soft_iou_f1"]["mean"], 0)

    def test_duplicate_penalty_and_missing_failure(self):
        p = structure()
        p["objects"].append(dict(p["objects"][0], id="b"))
        p["layer_order"].append("b")
        s = self.run_score(structure(), p)["primary"]
        self.assertAlmostEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], 2/3)
        s = self.run_score(structure(), None)["primary"]
        self.assertEqual(s["output_usability"]["first_pass_core_valid_rate"], 0)
        self.assertEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], 0)
        self.assertIsNone(s["localization"]["photo"]["soft_iou_f1"]["mean"])

    def test_stage_required_and_background_order_not_repaired(self):
        self.assertIsNone(self.run_score(structure(), structure(), "unspecified")["primary"]
                          ["output_usability"]["first_pass_core_valid_rate"])
        p = deepcopy(structure())
        p["background"] = {"type": "photographic", "id": "bg"}
        p["layer_order"].insert(0, "bg")
        self.assertEqual(self.run_score(structure(), p)["primary"]["output_usability"]
                         ["first_pass_core_valid_rate"], 0)

    def test_invalid_order_preserves_raw_geometry_credit(self):
        p = structure()
        p["layer_order"] = []
        s = self.run_score(structure(), p)["primary"]
        self.assertEqual(s["output_usability"]["first_pass_core_valid_rate"], 0)
        self.assertEqual(s["localization"]["fixed"]["soft_iou_f1"]["mean"], 1)

    def test_failed_final_not_dropped_from_mean(self):
        g = {k: Record(k, structure()) for k in ("a", "b")}
        s, _ = evaluate_dataset(g, {"a": Record("a", structure())}, prediction_stage="final_accepted")
        self.assertEqual(s["primary"]["output_usability"]["final_core_valid_rate"], .5)
        self.assertIsNone(s["primary"]["output_usability"]["first_pass_core_valid_rate"])
        self.assertEqual(s["primary"]["localization"]["fixed"]["soft_iou_f1"],
                         {"mean": .5, "eligible_images": 2, "total_images": 2})
        self.assertEqual(s["primary"]["treatment"]["kind_agreement_proxy"]["eligible_images"], 1)

    def test_failed_workflow_ignores_stale_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "cases" / "case" / "structure.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(structure()), encoding="utf-8")
            case = {"id": "case", "relative_path": "image.png", "status": "failed"}
            record, source = final_prediction("x", root, case)
            self.assertIsNone(record)
            self.assertTrue(source["missing"])
            case["status"] = "complete"
            self.assertIsNotNone(final_prediction("x", root, case)[0])


if __name__ == "__main__":
    unittest.main()
