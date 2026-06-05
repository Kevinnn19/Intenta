from app.preprocessing.text_cleaner import clean_text, clean_batch

texts = ["My PAYMENT failed!!!","Refund not received","   account locked   ","WHY is this happening???"]

print(clean_text("My PAYMENT failed!!!"))
print(clean_text("Refund not received👀"))
print(clean_text("   account locked   "))
print(clean_text("WHY is this happening???"))
print(clean_batch(texts))