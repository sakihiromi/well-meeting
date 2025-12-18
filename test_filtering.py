#!/usr/bin/env python3
# test_filtering.py
# フィルタリング機能のテストスクリプト

import sys
sys.path.insert(0, '.')

from mic_api import is_complete_sentence, should_filter_text, init_params

def test_mechanical_filter():
    """機械的フィルタのテスト"""
    print("=" * 60)
    print("機械的フィルタのテスト")
    print("=" * 60)
    
    test_cases = [
        # (text, expected_complete)
        ("今日の会議の議題は売上向上についてです", True),
        ("了解しました", True),
        ("それについては後で検討しましょう", True),
        ("ありがとうございます。", True),
        ("わかりました!", True),
        ("本当ですか?", True),
        ("今日の会議の議題は", False),
        ("それについては", False),
        ("えっと、あの", False),
        ("うーん", False),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = is_complete_sentence(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{text}' -> {result} (期待: {expected})")
    
    print(f"\n結果: {passed}件成功, {failed}件失敗\n")
    return failed == 0

def test_filter_with_params():
    """パラメータを使った総合的なフィルタテスト"""
    print("=" * 60)
    print("総合フィルタのテスト (LLMフィルタ無効)")
    print("=" * 60)
    
    # LLMフィルタを無効にしてテスト
    params = {
        "openai_api_key": "dummy",
        "enable_llm_filter": False,
        "filter_confidence_threshold": 0.6,
    }
    
    test_cases = [
        # (text, should_be_filtered)
        ("今日の会議の議題は売上向上についてです", False),
        ("了解", False),  # 5文字以上なので通る
        ("それについては後で検討しましょう", False),
        ("あ", True),  # 短すぎる
        ("うん", True),  # 短すぎる
        ("ええと", False),  # 5文字以上だが終助詞なし
        ("今日の会議の議題は", False),  # 5文字以上だが不完全（機械的には通る）
    ]
    
    passed = 0
    failed = 0
    
    for text, should_filter in test_cases:
        result, reason = should_filter_text(text, params)
        status = "✓" if result == should_filter else "✗"
        if result == should_filter:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{text}' -> フィルタ: {result}, 理由: {reason}")
        print(f"   期待: {'フィルタする' if should_filter else '通す'}")
    
    print(f"\n結果: {passed}件成功, {failed}件失敗\n")
    return failed == 0

def test_llm_filter_sample():
    """LLMフィルタのサンプルテスト（実際にAPIを呼ぶ）"""
    print("=" * 60)
    print("LLMフィルタのサンプルテスト")
    print("=" * 60)
    
    try:
        params = init_params(".env")
        
        if not params.get("openai_api_key"):
            print("OPENAI_API_KEY が設定されていないため、LLMフィルタテストをスキップします。")
            return True
        
        test_cases = [
            "今日の会議の議題は売上向上についてです",
            "今日の会議の議題は",
            "えっと、その",
            "了解しました",
        ]
        
        for text in test_cases:
            should_filter, reason = should_filter_text(text, params)
            print(f"'{text}'")
            print(f"  -> フィルタ: {should_filter}, 理由: {reason}\n")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 フィルタリング機能のテスト\n")
    
    success = True
    success = test_mechanical_filter() and success
    success = test_filter_with_params() and success
    
    # LLMフィルタのテストは参考程度
    print("\n" + "=" * 60)
    print("参考: LLMフィルタのテスト（実際にAPIを呼び出します）")
    print("=" * 60)
    test_llm_filter_sample()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ すべての基本テストが成功しました！")
    else:
        print("❌ 一部のテストが失敗しました。")
    print("=" * 60)
