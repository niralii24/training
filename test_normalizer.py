from text_normalizer import normalize_all_candidates

# Your real candidate transcripts (Arabic)
candidates = [
    "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف ، ليه اليكس ارسلى ابيك بكلمة؟ بالظبط . صح؟ ويش يقصد بها؟ ابيك بكلمة. عن ويش؟ وليه؟ متى ؟وين؟ وعن ويش",
    
    "ااااه ثيو يا الله لازم تشوف ذا الحين , لا اخس شوف , ليه اليكس ارسلى ابيك بكلمة,  بالظبط  صح ويش يقصد بها ابيك بكلمة عن ويش وليه متى وين وعن ويش",
    
    "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف , ليه اليكس ارسلى ابيك بكلمة؟  بالظبط . صح؟ ويش يقصد بها؟ ابيك بكلمة. عن ويش؟ وليه؟ متى ؟وين؟ وعن ويش",
    
    "ااااه. ثيو يا الله. لازم تشوف ذا الحين . لا اخس شوف ، ليه اليكس ارسلى ابيك بكلمة؟ بالظبط . صح؟ ويش يقصد بها؟ ابيك بكلمة. عن ويش؟ وليه؟ متى ؟وين؟ وعن ويش"
]

# Normalize all of them as Arabic
normalized = normalize_all_candidates(candidates, language="ar")

print("\n--- Final Normalized Candidates ---")
for i, text in enumerate(normalized):
    print(f"Candidate {i+1}: {text}")

print("\n--- Comparing candidates ---")
# Check which ones became identical after normalization
for i in range(len(normalized)):
    for j in range(i+1, len(normalized)):
        if normalized[i] == normalized[j]:
            print(f"Candidate {i+1} and Candidate {j+1} are IDENTICAL after normalization ✅")
        else:
            print(f"Candidate {i+1} and Candidate {j+1} are DIFFERENT ⚠️")