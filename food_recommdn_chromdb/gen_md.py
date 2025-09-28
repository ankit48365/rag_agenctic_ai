# to generate dynamic markdown files for the food recommendation database
import pdoc

doc = pdoc.pdoc("interactive_search")
with open("food_recommendation.md", "w", encoding="utf-8") as f:
    f.write(doc)