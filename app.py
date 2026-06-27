from graph import app

result = app.invoke({
    "question": "Quel est le bénéfice net en 2024 ?"
})

print("\nFINAL ANSWER:\n")
print(result["answer"])