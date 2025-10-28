#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the two-step recommendation system.
This script demonstrates how to use the MaterialRecommendationSystem class.
"""

from recommendation import MaterialRecommendationSystem
from pathlib import Path
import json


def example_1_basic_usage():
    """
    Example 1: Basic usage with a single question.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Recommendation")
    print("="*80)
    
    # Initialize the recommendation system
    # It will automatically use settings from config.yml
    recommender = MaterialRecommendationSystem()
    
    # Define a question and the student's wrong answer
    question = """
        A truck is pulling a car. Let FT be the magnitude of the force that the
    truck exerts on the car and let FC be the magnitude of the force that the car exerts on the truck.
    If the truck is speeding up while driving up a mountain and friction is negligible, how does the
    magnitude of FT compare to FC?
    A. FT > FC
    B. FC > FT
    C. FT = FC = 0
    D. FT = FC > 0
    """
    
    wrong_answer = "C. FT = FC = 0"
    correct_answer = "D. FT = FC > 0"
    
    # Get recommendation
    try:
        recommendation = recommender.recommend(
            question=question,
            wrong_answer=wrong_answer,
            correct_answer=correct_answer,
            verbose=True
        )
        
        print("\n✓ Recommendation completed successfully!")
        
        # Save to JSON file
        output_file = Path("example_recommendation_1.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recommendation, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    """Run example 1: Basic recommendation."""
    print("\n" + "="*80)
    print("RECOMMENDATION SYSTEM - EXAMPLE 1")
    print("="*80)
    print("\nThis script demonstrates basic usage of the recommendation system.")
    print("Make sure you have:")
    print("  1. Processed materials in the database (materials/processed_materials.db)")
    print("  2. An LLM server running (or configure API endpoint in config.yml)")
    print("  3. Generated descriptions for materials (run materials_description_gen.py)")
    
    example_1_basic_usage()


if __name__ == "__main__":
    main()

