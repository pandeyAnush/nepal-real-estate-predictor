#!/usr/bin/env python3
"""
PRODUCTION CLEANING - ALL REAL ESTATE CATEGORIES
House + Land + Apartment dataset
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

class AllCategoriesRealEstateCleaner:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self.log = []
        
        print("\n" + "="*100)
        print("PRODUCTION CLEANING - ALL REAL ESTATE CATEGORIES")
        print("="*100)
        
        print(f"\n INPUT: {input_file}")
        print(f"   Rows: {len(self.df)}")
        print(f"   Columns: {len(self.df.columns)}")
    
    def log_action(self, action, count, reason):
        """Log cleaning actions"""
        msg = f"   [{action}] {count} rows — {reason}"
        print(msg)
        self.log.append(msg)
    
    #  STEP 1: HANDLE MISSING LOCATION 
    '''Fix Missing Locations
       If a row has no location info, it grabs the district name instead.
       If even that's missing, it writes "Unknown". Nobody gets left blank.'''
    
    def clean_location(self):
        """Handle missing locations"""
        print(f"\n{'='*100}")
        print("STEP 1: HANDLE MISSING LOCATION")
        print("="*100)
        
        missing_loc = self.df['location'].isna().sum()
        print(f"\n   Missing location: {missing_loc} rows")
        
        if missing_loc > 0:
            # Try to extract from district_clean if available
            for idx in self.df[self.df['location'].isna()].index:
                if pd.notna(self.df.loc[idx, 'district_clean']):
                    self.df.loc[idx, 'location'] = self.df.loc[idx, 'district_clean']
                else:
                    self.df.loc[idx, 'location'] = 'Unknown'
            
            self.log_action("LOCATION_FILLED", missing_loc, "Filled from district or marked Unknown")
        
        print(f"   ✅ All locations now populated")
    
    #  STEP 2: VALIDATE PRICE & AREA 
    ''' Remove Bad Prices & Areas
        Throws out any row where price is 0, negative,
        or missing — and same for land area. 
        A property with no price or no size is useless for prediction.'''
    
    def validate_price_area(self):
        """Remove invalid price/area"""
        print(f"\n{'='*100}")
        print("STEP 2: VALIDATE PRICE & AREA")
        print("="*100)
        
        before = len(self.df)
        
        # Remove zero or negative prices
        invalid_price = (self.df['price_numeric'] <= 0) | (self.df['price_numeric'].isna())
        
        # Remove zero or negative areas
        invalid_area = (self.df['land_area_sqft'] <= 0) | (self.df['land_area_sqft'].isna())
        
        removed = (invalid_price | invalid_area).sum()
        
        self.df = self.df[~(invalid_price | invalid_area)]
        
        print(f"\n   Before: {before} rows")
        print(f"   Removed (invalid price/area): {removed}")
        print(f"   After: {len(self.df)} rows")
        
        self.log_action("INVALID_PRICE_AREA", removed, "Price ≤ 0 or Area ≤ 0")
    
    #  STEP 3: HANDLE BEDROOMS/BATHROOMS 
    ''' Fill in Missing Bedrooms & Bathrooms
        For rows missing bedrooms or bathrooms, 
        it calculates the median for that property category 
        (House/Land/Apartment) and fills it in. 
        Land plots get 0 by default since they have no rooms.'''
    

    def impute_rooms(self):
        """Impute missing bedrooms/bathrooms"""
        print(f"\n{'='*100}")
        print("STEP 3: IMPUTE BEDROOMS & BATHROOMS")
        print("="*100)
        
        print(f"\n   Before imputation:")
        print(f"      Bedrooms missing: {self.df['bedrooms'].isna().sum()}")
        print(f"      Bathrooms missing: {self.df['bathrooms'].isna().sum()}")
        
        # For each category, impute with category median
        for category in self.df['category'].unique():
            mask_cat = self.df['category'] == category
            
            # Bedrooms
            mask_bed = mask_cat & self.df['bedrooms'].isna()
            if mask_bed.sum() > 0:
                median = self.df[mask_cat]['bedrooms'].median()
                if pd.isna(median):
                    median = self.df['bedrooms'].median()
                if pd.isna(median):
                    median = 0  # Default for Land
                self.df.loc[mask_bed, 'bedrooms'] = median
            
            # Bathrooms
            mask_bath = mask_cat & self.df['bathrooms'].isna()
            if mask_bath.sum() > 0:
                median = self.df[mask_cat]['bathrooms'].median()
                if pd.isna(median):
                    median = self.df['bathrooms'].median()
                if pd.isna(median):
                    median = 0  # Default for Land
                self.df.loc[mask_bath, 'bathrooms'] = median
        
        print(f"\n   After imputation:")
        print(f"      Bedrooms missing: {self.df['bedrooms'].isna().sum()}")
        print(f"      Bathrooms missing: {self.df['bathrooms'].isna().sum()}")
    
    #  STEP 4: RECALCULATE MISSING ENGINEERED FEATURES 

    ''' Rebuild Calculated Columns
        Recalculates three columns fresh:
        price_per_sqft = price ÷ area
        bed_bath_ratio = bedrooms ÷ (bathrooms + 0.1) — the 0.1 prevents divide-by-zero
        is_bungalow, is_villa, mentions_bhk — scans the title text and puts 1 or 0 '''
    

    def recalculate_engineered_features(self):
        """Recalculate missing engineered features"""
        print(f"\n{'='*100}")
        print("STEP 4: RECALCULATE ENGINEERED FEATURES")
        print("="*100)
        
        # price_per_sqft
        print(f"\n   Recalculating price_per_sqft...")
        self.df['price_per_sqft'] = self.df['price_numeric'] / self.df['land_area_sqft']
        
        # bed_bath_ratio
        print(f"   Recalculating bed_bath_ratio...")
        self.df['bed_bath_ratio'] = self.df['bedrooms'] / (self.df['bathrooms'] + 0.1)
        
        # Text features from title
        print(f"   Extracting text features from title...")
        if 'title' in self.df.columns:
            text = self.df['title'].fillna('').str.lower()
            self.df['is_bungalow'] = text.str.contains('bungalow', na=False).astype(int)
            self.df['is_villa'] = text.str.contains('villa', na=False).astype(int)
            self.df['mentions_bhk'] = text.str.contains('bhk|b\.h\.k', na=False).astype(int)
        
        print(f"   ✅ All engineered features recalculated")
    
    #  STEP 5: VALIDATE CATEGORIES 
    ''' Makes sure all category names are spelled the same way 
        (e.g., "flat" and "Flat" both become "Apartment"). 
        Rows with unrecognizable categories get removed.'''



    def validate_categories(self):
        """Validate property categories"""
        print(f"\n{'='*100}")
        print("STEP 5: VALIDATE CATEGORIES")
        print("="*100)
        
        print(f"\n   Category distribution BEFORE:")
        print(self.df['category'].value_counts().to_string())
        
        # Standardize categories
        category_map = {
            'House': 'House',
            'house': 'House',
            'Land': 'Land',
            'land': 'Land',
            'Apartment': 'Apartment',
            'apartment': 'Apartment',
            'Flat': 'Apartment',
            'flat': 'Apartment',
        }
        
        self.df['category'] = self.df['category'].map(lambda x: category_map.get(x, 'Other'))
        
        # Remove 'Other' if any
        other_count = (self.df['category'] == 'Other').sum()
        if other_count > 0:
            self.df = self.df[self.df['category'] != 'Other']
            self.log_action("REMOVE_OTHER", other_count, "Ambiguous category")
        
        print(f"\n   Category distribution AFTER:")
        print(self.df['category'].value_counts().to_string())
    
    #  STEP 6: REMOVE DUPLICATES 
    ''' Remove Duplicates
        First removes exact duplicate rows, 
        then removes near-duplicates — listings that have the 
        same title + location + price + category even 
        if other columns differ slightly. '''
    
    def remove_duplicates(self):
        print(f"\n{'='*100}")
        print("STEP 6: REMOVE DUPLICATES")
        print("="*100)
        
        before = len(self.df)
        
        # Exact duplicates
        self.df = self.df.drop_duplicates()
        exact_removed = before - len(self.df)
        
        # Near-duplicates (same title, location, price, category)
        before_near = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=['title', 'location', 'price_numeric', 'category'],
            keep='first'
        )
        near_removed = before_near - len(self.df)
        
        print(f"\n   Exact duplicates removed: {exact_removed}")
        print(f"   Near-duplicates removed: {near_removed}")
        print(f"   Total removed: {exact_removed + near_removed}")
        print(f"   Final: {len(self.df)} rows")
    
    #  STEP 7: FINAL VALIDATION 

    ''' Final Check
        Scans the 5 most critical columns 
        (title, location, category, price, area) 
        and drops any remaining rows still missing values.'''
    
    def final_validation(self):
        print(f"\n{'='*100}")
        print("STEP 7: FINAL VALIDATION")
        print("="*100)
        
        critical_cols = ['title', 'location', 'category', 'price_numeric', 'land_area_sqft']
        
        print(f"\n   Missing values in critical columns:")
        for col in critical_cols:
            missing = self.df[col].isna().sum()
            if missing > 0:
                print(f"        {col}: {missing}")
            else:
                print(f"       {col}: 0")
        
        # Remove any remaining rows with critical missing values
        before_val = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        after_val = len(self.df)
        
        if before_val > after_val:
            print(f"\n   Removed {before_val - after_val} rows with critical missing values")
    
    #  STEP 8: PRINT FINAL STATISTICS 
    '''Saves only the 14 most useful columns 
    to cleaned_realestate_all_categories_final.csv, 
    dropping scrape metadata like URLs and dates.'''

    def print_final_statistics(self):
        print(f"\n{'='*100}")
        print(" FINAL DATASET STATISTICS")
        print("="*100)
        
        print(f"\n TOTAL PROPERTIES: {len(self.df)}")
        
        print(f"\n PROPERTY CATEGORIES:")
        cat_counts = self.df['category'].value_counts()
        total = len(self.df)
        for cat, count in cat_counts.items():
            pct = count / total * 100
            print(f"   {cat:15s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n📍 TOP DISTRICTS:")
        dist_counts = self.df['district_clean'].value_counts().head(10)
        for dist, count in dist_counts.items():
            pct = count / total * 100
            print(f"   {dist:30s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n PRICE STATISTICS:")
        print(f"   Min:    ₨{self.df['price_numeric'].min():,.0f}")
        print(f"   Q1:     ₨{self.df['price_numeric'].quantile(0.25):,.0f}")
        print(f"   Median: ₨{self.df['price_numeric'].median():,.0f}")
        print(f"   Q3:     ₨{self.df['price_numeric'].quantile(0.75):,.0f}")
        print(f"   Max:    ₨{self.df['price_numeric'].max():,.0f}")
        print(f"   Mean:   ₨{self.df['price_numeric'].mean():,.0f}")
        
        print(f"\n LAND AREA STATISTICS:")
        print(f"   Min:    {self.df['land_area_sqft'].min():,.0f} sq ft")
        print(f"   Median: {self.df['land_area_sqft'].median():,.0f} sq ft")
        print(f"   Mean:   {self.df['land_area_sqft'].mean():,.0f} sq ft")
        print(f"   Max:    {self.df['land_area_sqft'].max():,.0f} sq ft")
        
        print(f"\n DATA SOURCES:")
        if 'source' in self.df.columns:
            print(self.df['source'].value_counts().to_string())
    
    #  SAVE FINAL DATASET 
    '''It's a 7-step pipeline that goes from raw messy data → removes garbage rows → 
    fills gaps → standardizes labels → kills duplicates → 
    saves a clean file ready for your ML model.'''
        
    def save_final(self, output_file='cleaned_realestate_all_categories_final.csv'):
        """Save production-ready dataset"""
        # Select key columns
        key_cols = ['title', 'location', 'district_clean', 'category',
                   'price_numeric', 'land_area_sqft', 'bedrooms', 'bathrooms',
                   'price_per_sqft', 'bed_bath_ratio', 'is_bungalow', 'is_villa',
                   'mentions_bhk', 'source']
        
        available_cols = [c for c in key_cols if c in self.df.columns]
        df_final = self.df[available_cols].copy()
        
        df_final.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n{'='*100}")
        print(f" PRODUCTION-READY FILE SAVED")
        print(f"{'='*100}")
        print(f"\n File: {output_file}")
        print(f" Rows: {len(df_final)}")
        print(f" Columns: {len(df_final.columns)}")
        print(f" Missing values: 0")
        print(f" Status: READY FOR WEEK 3 NLP\n")
        
        return df_final
    
    def run_pipeline(self):
        """Execute complete cleaning pipeline"""
        self.clean_location()
        self.validate_price_area()
        self.impute_rooms()
        self.recalculate_engineered_features()
        self.validate_categories()
        self.remove_duplicates()
        self.final_validation()
        self.print_final_statistics()
        return self.save_final()

# Run
if __name__ == "__main__":
    cleaner = AllCategoriesRealEstateCleaner('cleaned_properties_nepal_all_categories.csv')
    df_final = cleaner.run_pipeline()
    
    print("="*100)
    print(" Data cleaning pipeline completed successfully!")
    print("="*100 + "\n")