import json
import subprocess
import sys
from pathlib import Path

import pytest

from forge.chat import Assistant
from forge.robot import SPICE_SLOTS, assess
from forge.search import Engine

ROOT = Path(__file__).resolve().parent.parent
assistant = Assistant(Engine())


def rec(name, method="Heat oil. Add the vegetables and stir for 10 minutes.", ids=()):
    return {"name": name, "instructions": method, "ingredient_ids": list(ids), "servings": 4}


@pytest.mark.parametrize("name,why", [
    ("Aloo Paratha", "flipping"), ("Masala Dosa", "flipping"), ("Ragi Ujju Rotti", "flipping"),
    ("Onion Pakora", "deep-frying"), ("Kanchipuram Idlis", "steamer"), ("Khaman Dhokla", "steamer"),
    ("Cucumber Raita", "no cooking"), ("Eggless Chocolate Cake", "oven"),
])
def test_dishes_the_robot_cannot_cook(name, why):
    a = assess(rec(name))
    assert a["fit"] == "manual" and why in a["reason"]


@pytest.mark.parametrize("name", ["Dal Tadka", "Paneer Butter Masala", "Moong Dal Khichdi", "Rice Kheer", "Rajma Masala"])
def test_one_pot_dishes_are_cookable(name):
    assert assess(rec(name))["fit"] == "robot"


def test_method_frying_marks_manual_but_dish_name_wins():
    assert assess(rec("Aloo Bites", "Deep fry the balls in hot oil until golden."))["reason"] == "needs deep-frying"
    assert "steamer" in assess(rec("Pudina Idli", "Steam the idlis. Optionally deep fry leftovers."))["reason"]


def test_prep_steps_and_spice_slots():
    a = assess(rec("Chana Masala", "Soak the chickpeas overnight. Chop the onions. Heat oil and add the spices.",
                   ids=["cumin", "turmeric_powder", "red_chilli", "garam_masala", "coriander_seed", "salt", "amchur",
                        "black_pepper", "cinnamon", "cloves"]))
    assert a["prep"] == ["Soak the chickpeas overnight.", "Chop the onions."]
    assert a["extra_spices"] == 10 - SPICE_SLOTS


def test_catalogue_robot_data_overrides_rules():
    r = rec("Masala Dosa")
    r["robot"] = {"fit": "robot", "reason": "", "prep": ["Make the batter"], "prep_min": 5}
    assert assess(r)["fit"] == "robot"


def test_app_shows_only_cookable_dishes_and_explains():
    out = assistant.chat(None, "masala dosa")
    assert all(c["robot"]["fit"] == "robot" for c in out["results"])
    assert "can't make masala dosa" in out["reply"]
    assert any(c["kind"] == "robot" for c in out["constraints"])


def test_robot_chip_can_be_removed():
    state = assistant.chat(None, "masala dosa")["state"]
    s = assistant.remove_constraint(state, "robot", "1")
    out = assistant.chat(s.to_dict(), "masala dosa")
    assert any("dosa" in c["name"].lower() for c in out["results"])


def test_big_group_gets_batch_note():
    out = assistant.chat(None, "jain dinner for 10")
    assert any("batches" in w for c in out["results"] for w in c["why"])


def test_import_sample_catalogue(tmp_path):
    out = tmp_path / "cat"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "import_catalogue.py"),
                    str(ROOT / "data" / "catalogues" / "sample_nosh.json"), str(out)], check=True,
                   capture_output=True, cwd=ROOT)
    recipes = [json.loads(l) for l in (out / "recipes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recipes) == 3
    a = Assistant(Engine(data_dir=out))
    names = [c["name"] for c in a.chat(None, "dinner")["results"]]
    assert "SAMPLE Phulka" not in names and "SAMPLE Moong Dal Khichdi" in names
    jain = [c["name"] for c in a.chat(None, "jain dinner")["results"]]
    assert "SAMPLE Jeera Aloo" not in jain  # potato isn't Jain: diet checks run on imported recipes too


def test_steps_split_in_order_prep_robot_finish():
    from forge.robot import split_steps
    steps = ["Prepare cornstarch paste by mixing cornstarch in cold water.", "Mix condensed milk and milk powder.",
             "Take mango puree in a non stick saucepan kept on medium heat.", "Add the paste and mix well.",
             "Let it cook until it thickens.", "Once done turn off the heat.",
             "Grease your palms with ghee and shape into balls.", "Serve as a dessert."]
    prep, robot, finish = split_steps(steps)
    assert prep == steps[:2] and robot == steps[2:6] and finish == steps[6:]


def test_pressure_cooking_before_the_pan_is_prep():
    from forge.robot import split_steps
    steps = ["Pressure cook the raw bananas for 2 whistles.", "Turn off the heat and allow the pressure to release.",
             "Heat oil in a kadai.", "Add cumin seeds and the bananas.", "Cook for 5 minutes.", "Serve hot."]
    prep, robot, finish = split_steps(steps)
    assert prep == steps[:2] and robot == steps[2:5] and finish == steps[5:]
