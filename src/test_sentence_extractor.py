#!/usr/bin/env python3
"""
Test file for the extract_sentences function in utils.py
"""

from src.utils import extract_sentences


def test_extract_sentences():
    """Test the extract_sentences function with various examples."""
    
    print("Testing extract_sentences function...\n")
    
    # Test case 1: Basic sentence splitting with longer sentences
    test1 = "Hello world! How are you today? I am doing very well"
    result1 = extract_sentences(test1)
    print(f"Test 1 - Basic splitting:")
    print(f"Input: {test1}")
    print(f"Output: {result1}")
    print(f"Expected: ['Hello world!', 'How are you today?', 'I am doing very well.']")
    print()
    
    # Test case 2: Multiple punctuation marks with longer sentences
    test2 = "What do you think?! Really, is that so?!! Yes, I believe it is."
    result2 = extract_sentences(test2)
    print(f"Test 2 - Multiple punctuation:")
    print(f"Input: {test2}")
    print(f"Output: {result2}")
    print(f"Expected: ['What do you think?!', 'Really, is that so?!!', 'Yes, I believe it is.']")
    print()
    
    # Test case 3: Short sentences that should be merged
    test3 = "Short. This is a longer sentence. Yes."
    result3 = extract_sentences(test3)
    print(f"Test 3 - Short sentence merging:")
    print(f"Input: {test3}")
    print(f"Output: {result3}")
    print(f"Expected: ['Short. This is a longer sentence. Yes.']")
    print()
    
    # Test case 4: Very short sentences
    test4 = "Hi. OK. No. Maybe."
    result4 = extract_sentences(test4)
    print(f"Test 4 - Very short sentences:")
    print(f"Input: {test4}")
    print(f"Output: {result4}")
    print(f"Expected: ['Hi. OK. No. Maybe.']")
    print()
    
    # Test case 5: Empty and whitespace sentences
    test5 = "First sentence.   . Second sentence.   "
    result5 = extract_sentences(test5)
    print(f"Test 5 - Empty sentences:")
    print(f"Input: {test5}")
    print(f"Output: {result5}")
    print(f"Expected: ['First sentence.', 'Second sentence.']")
    print()
    
    # Test case 6: Complex punctuation
    test6 = "Wait... What??! No way!!! Really?"
    result6 = extract_sentences(test6)
    print(f"Test 6 - Complex punctuation:")
    print(f"Input: {test6}")
    print(f"Output: {result6}")
    print(f"Expected: ['Wait...', 'What??!', 'No way!!!', 'Really?']")
    print()
    
    # Test case 7: Long text with mixed sentence lengths
    test7 = "This is a very long sentence that should remain as is. Short. Another long sentence with multiple clauses and complex structure. OK. Final sentence here."
    result7 = extract_sentences(test7)
    print(f"Test 7 - Mixed sentence lengths:")
    print(f"Input: {test7}")
    print(f"Output: {result7}")
    print(f"Expected: ['This is a very long sentence that should remain as is. Short.', 'Another long sentence with multiple clauses and complex structure. OK. Final sentence here.']")
    print()
    
    # Test case 8: No punctuation
    test8 = "This is just one long sentence without any punctuation"
    result8 = extract_sentences(test8)
    print(f"Test 8 - No punctuation:")
    print(f"Input: {test8}")
    print(f"Output: {result8}")
    print(f"Expected: ['This is just one long sentence without any punctuation']")
    print()
    
    # Test case 9: Empty string
    test9 = ""
    result9 = extract_sentences(test9)
    print(f"Test 9 - Empty string:")
    print(f"Input: '{test9}'")
    print(f"Output: {result9}")
    print(f"Expected: []")
    print()
    
    # Test case 10: Only punctuation
    test10 = "!!! ??? ..."
    result10 = extract_sentences(test10)
    print(f"Test 10 - Only punctuation:")
    print(f"Input: {test10}")
    print(f"Output: {result10}")
    print(f"Expected: []")
    print()
    
    # Test case 11: Whitespace only
    test11 = "   \n\t   "
    result11 = extract_sentences(test11)
    print(f"Test 11 - Whitespace only:")
    print(f"Input: '{test11}'")
    print(f"Output: {result11}")
    print(f"Expected: []")
    print()
    
    # Test case 12: Single word sentences
    test12 = "Yes. No. Maybe. Sure."
    result12 = extract_sentences(test12)
    print(f"Test 12 - Single word sentences:")
    print(f"Input: {test12}")
    print(f"Output: {result12}")
    print(f"Expected: ['Yes. No. Maybe. Sure.']")
    print()
    
    # Test case 13: Real-world example
    test13 = "The quick brown fox jumps over the lazy dog. Wow! Is that true? Absolutely not. The dog is actually quite active."
    result13 = extract_sentences(test13)
    print(f"Test 13 - Real-world example:")
    print(f"Input: {test13}")
    print(f"Output: {result13}")
    print(f"Expected: ['The quick brown fox jumps over the lazy dog.', 'Wow!', 'Is that true?', 'Absolutely not. The dog is actually quite active.']")
    print()
    
    # Test case 14: Original failing case to show merging behavior
    test14 = "Hello world! How are you? Fine."
    result14 = extract_sentences(test14)
    print(f"Test 14 - Original case (shows merging):")
    print(f"Input: {test14}")
    print(f"Output: {result14}")
    print(f"Expected: ['Hello world! How are you? Fine.'] (all merged due to short sentences)")
    print()


def run_all_tests():
    """Run all test cases and provide a summary."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE FOR extract_sentences FUNCTION")
    print("=" * 80)
    print()
    
    test_extract_sentences()
    
    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
