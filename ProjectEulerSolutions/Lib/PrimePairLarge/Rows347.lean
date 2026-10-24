import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11549_small : List Nat := []
theorem row11549_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11549 row11549_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11549_block_005 : List Nat := []
theorem row11549_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11549 row11549_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11549_block_006 : List Nat := [11783, 12107]
def plan11549_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 19), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 31), .first, .first, .second, .first, (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row11549_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11549 row11549_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11549_block_006
  · decide
  · rfl

def row11549_block_007 : List Nat := [12917, 13577]
def plan11549_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .second, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 29), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11549_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11549 row11549_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11549_block_007
  · decide
  · rfl

def row11549_block_008 : List Nat := []
def plan11549_block_008 : List Budget.Choice := [.first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11549_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11549 row11549_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11549_block_008
  · decide
  · rfl

def row11549_block_009 : List Nat := []
theorem row11549_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11549 row11549_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11549_block_010 : List Nat := []
theorem row11549_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11549 row11549_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11549_block_011 : List Nat := []
theorem row11549_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11549 row11549_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11549_block_012 : List Nat := []
theorem row11549_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11549 row11549_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11549_block_013 : List Nat := []
theorem row11549_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11549 row11549_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11549_large_left_left : List Nat := row11549_small ++ row11549_block_005
theorem row11549_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11549 row11549_large_left_left :=
  Budget.RowUpperBound.branch row11549_small_covers row11549_block_005_covers

def row11549_large_left_right_right : List Nat := row11549_block_007 ++ row11549_block_008
theorem row11549_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11549 row11549_large_left_right_right :=
  Budget.RowUpperBound.branch row11549_block_007_covers row11549_block_008_covers

def row11549_large_left_right : List Nat := row11549_block_006 ++ row11549_large_left_right_right
theorem row11549_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11549 row11549_large_left_right :=
  Budget.RowUpperBound.branch row11549_block_006_covers row11549_large_left_right_right_covers

def row11549_large_left : List Nat := row11549_large_left_left ++ row11549_large_left_right
theorem row11549_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11549 row11549_large_left :=
  Budget.RowUpperBound.branch row11549_large_left_left_covers row11549_large_left_right_covers

def row11549_large_right_left : List Nat := row11549_block_009 ++ row11549_block_010
theorem row11549_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11549 row11549_large_right_left :=
  Budget.RowUpperBound.branch row11549_block_009_covers row11549_block_010_covers

def row11549_large_right_right_right : List Nat := row11549_block_012 ++ row11549_block_013
theorem row11549_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11549 row11549_large_right_right_right :=
  Budget.RowUpperBound.branch row11549_block_012_covers row11549_block_013_covers

def row11549_large_right_right : List Nat := row11549_block_011 ++ row11549_large_right_right_right
theorem row11549_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11549 row11549_large_right_right :=
  Budget.RowUpperBound.branch row11549_block_011_covers row11549_large_right_right_right_covers

def row11549_large_right : List Nat := row11549_large_right_left ++ row11549_large_right_right
theorem row11549_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11549 row11549_large_right :=
  Budget.RowUpperBound.branch row11549_large_right_left_covers row11549_large_right_right_covers

def row11549_large : List Nat := row11549_large_left ++ row11549_large_right
theorem row11549_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11549 row11549_large :=
  Budget.RowUpperBound.branch row11549_large_left_covers row11549_large_right_covers

def row11549 : List Nat := row11549_large
theorem row11549_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11549 row11549 := row11549_large_covers

def row11551_small : List Nat := []
theorem row11551_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11551 row11551_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11551_block_005 : List Nat := []
theorem row11551_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11551 row11551_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11551_block_006 : List Nat := [11833, 11887]
def plan11551_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .second, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row11551_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11551 row11551_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11551_block_006
  · decide
  · rfl

def row11551_block_007 : List Nat := [13147, 14173, 14251]
def plan11551_block_007 : List Budget.Choice := [(.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .second, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11551_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11551 row11551_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11551_block_007
  · decide
  · rfl

def row11551_block_008 : List Nat := []
def plan11551_block_008 : List Budget.Choice := [.first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11551_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11551 row11551_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11551_block_008
  · decide
  · rfl

def row11551_block_009 : List Nat := []
theorem row11551_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11551 row11551_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11551_block_010 : List Nat := []
theorem row11551_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11551 row11551_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11551_block_011 : List Nat := []
theorem row11551_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11551 row11551_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11551_block_012 : List Nat := []
theorem row11551_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11551 row11551_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11551_block_013 : List Nat := []
theorem row11551_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11551 row11551_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11551_large_left_left : List Nat := row11551_small ++ row11551_block_005
theorem row11551_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11551 row11551_large_left_left :=
  Budget.RowUpperBound.branch row11551_small_covers row11551_block_005_covers

def row11551_large_left_right_right : List Nat := row11551_block_007 ++ row11551_block_008
theorem row11551_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11551 row11551_large_left_right_right :=
  Budget.RowUpperBound.branch row11551_block_007_covers row11551_block_008_covers

def row11551_large_left_right : List Nat := row11551_block_006 ++ row11551_large_left_right_right
theorem row11551_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11551 row11551_large_left_right :=
  Budget.RowUpperBound.branch row11551_block_006_covers row11551_large_left_right_right_covers

def row11551_large_left : List Nat := row11551_large_left_left ++ row11551_large_left_right
theorem row11551_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11551 row11551_large_left :=
  Budget.RowUpperBound.branch row11551_large_left_left_covers row11551_large_left_right_covers

def row11551_large_right_left : List Nat := row11551_block_009 ++ row11551_block_010
theorem row11551_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11551 row11551_large_right_left :=
  Budget.RowUpperBound.branch row11551_block_009_covers row11551_block_010_covers

def row11551_large_right_right_right : List Nat := row11551_block_012 ++ row11551_block_013
theorem row11551_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11551 row11551_large_right_right_right :=
  Budget.RowUpperBound.branch row11551_block_012_covers row11551_block_013_covers

def row11551_large_right_right : List Nat := row11551_block_011 ++ row11551_large_right_right_right
theorem row11551_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11551 row11551_large_right_right :=
  Budget.RowUpperBound.branch row11551_block_011_covers row11551_large_right_right_right_covers

def row11551_large_right : List Nat := row11551_large_right_left ++ row11551_large_right_right
theorem row11551_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11551 row11551_large_right :=
  Budget.RowUpperBound.branch row11551_large_right_left_covers row11551_large_right_right_covers

def row11551_large : List Nat := row11551_large_left ++ row11551_large_right
theorem row11551_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11551 row11551_large :=
  Budget.RowUpperBound.branch row11551_large_left_covers row11551_large_right_covers

def row11551 : List Nat := row11551_large
theorem row11551_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11551 row11551 := row11551_large_covers

def row11579_small : List Nat := []
theorem row11579_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11579 row11579_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11579_block_005 : List Nat := []
theorem row11579_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11579 row11579_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11579_block_006 : List Nat := [11789, 12011, 12263]
def plan11579_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.secondFactor 13), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 17), .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11579_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11579 row11579_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11579_block_006
  · decide
  · rfl

def row11579_block_007 : List Nat := [12893]
def plan11579_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 17), .second, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), .first, .second, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 31), .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .outside]
set_option maxRecDepth 2048 in
theorem row11579_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11579 row11579_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11579_block_007
  · decide
  · rfl

def row11579_block_008 : List Nat := []
theorem row11579_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11579 row11579_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11579_block_009 : List Nat := []
theorem row11579_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11579 row11579_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11579_block_010 : List Nat := []
theorem row11579_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11579 row11579_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11579_block_011 : List Nat := []
theorem row11579_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11579 row11579_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11579_block_012 : List Nat := []
theorem row11579_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11579 row11579_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11579_block_013 : List Nat := []
theorem row11579_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11579 row11579_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11579_large_left_left : List Nat := row11579_small ++ row11579_block_005
theorem row11579_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11579 row11579_large_left_left :=
  Budget.RowUpperBound.branch row11579_small_covers row11579_block_005_covers

def row11579_large_left_right_right : List Nat := row11579_block_007 ++ row11579_block_008
theorem row11579_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11579 row11579_large_left_right_right :=
  Budget.RowUpperBound.branch row11579_block_007_covers row11579_block_008_covers

def row11579_large_left_right : List Nat := row11579_block_006 ++ row11579_large_left_right_right
theorem row11579_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11579 row11579_large_left_right :=
  Budget.RowUpperBound.branch row11579_block_006_covers row11579_large_left_right_right_covers

def row11579_large_left : List Nat := row11579_large_left_left ++ row11579_large_left_right
theorem row11579_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11579 row11579_large_left :=
  Budget.RowUpperBound.branch row11579_large_left_left_covers row11579_large_left_right_covers

def row11579_large_right_left : List Nat := row11579_block_009 ++ row11579_block_010
theorem row11579_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11579 row11579_large_right_left :=
  Budget.RowUpperBound.branch row11579_block_009_covers row11579_block_010_covers

def row11579_large_right_right_right : List Nat := row11579_block_012 ++ row11579_block_013
theorem row11579_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11579 row11579_large_right_right_right :=
  Budget.RowUpperBound.branch row11579_block_012_covers row11579_block_013_covers

def row11579_large_right_right : List Nat := row11579_block_011 ++ row11579_large_right_right_right
theorem row11579_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11579 row11579_large_right_right :=
  Budget.RowUpperBound.branch row11579_block_011_covers row11579_large_right_right_right_covers

def row11579_large_right : List Nat := row11579_large_right_left ++ row11579_large_right_right
theorem row11579_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11579 row11579_large_right :=
  Budget.RowUpperBound.branch row11579_large_right_left_covers row11579_large_right_right_covers

def row11579_large : List Nat := row11579_large_left ++ row11579_large_right
theorem row11579_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11579 row11579_large :=
  Budget.RowUpperBound.branch row11579_large_left_covers row11579_large_right_covers

def row11579 : List Nat := row11579_large
theorem row11579_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11579 row11579 := row11579_large_covers

def row11587_small : List Nat := []
theorem row11587_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11587 row11587_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11587_block_005 : List Nat := []
theorem row11587_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11587 row11587_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11587_block_006 : List Nat := [12007]
def plan11587_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row11587_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11587 row11587_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11587_block_006
  · decide
  · rfl

def row11587_block_007 : List Nat := [12589, 12889]
def plan11587_block_007 : List Budget.Choice := [.second, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .possible, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 19), .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, .second, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .second, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 17), .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11587_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11587 row11587_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11587_block_007
  · decide
  · rfl

def row11587_block_008 : List Nat := []
theorem row11587_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11587 row11587_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11587_block_009 : List Nat := []
theorem row11587_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11587 row11587_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11587_block_010 : List Nat := []
theorem row11587_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11587 row11587_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11587_block_011 : List Nat := []
theorem row11587_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11587 row11587_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11587_block_012 : List Nat := []
theorem row11587_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11587 row11587_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11587_block_013 : List Nat := []
theorem row11587_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11587 row11587_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11587_large_left_left : List Nat := row11587_small ++ row11587_block_005
theorem row11587_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11587 row11587_large_left_left :=
  Budget.RowUpperBound.branch row11587_small_covers row11587_block_005_covers

def row11587_large_left_right_right : List Nat := row11587_block_007 ++ row11587_block_008
theorem row11587_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11587 row11587_large_left_right_right :=
  Budget.RowUpperBound.branch row11587_block_007_covers row11587_block_008_covers

def row11587_large_left_right : List Nat := row11587_block_006 ++ row11587_large_left_right_right
theorem row11587_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11587 row11587_large_left_right :=
  Budget.RowUpperBound.branch row11587_block_006_covers row11587_large_left_right_right_covers

def row11587_large_left : List Nat := row11587_large_left_left ++ row11587_large_left_right
theorem row11587_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11587 row11587_large_left :=
  Budget.RowUpperBound.branch row11587_large_left_left_covers row11587_large_left_right_covers

def row11587_large_right_left : List Nat := row11587_block_009 ++ row11587_block_010
theorem row11587_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11587 row11587_large_right_left :=
  Budget.RowUpperBound.branch row11587_block_009_covers row11587_block_010_covers

def row11587_large_right_right_right : List Nat := row11587_block_012 ++ row11587_block_013
theorem row11587_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11587 row11587_large_right_right_right :=
  Budget.RowUpperBound.branch row11587_block_012_covers row11587_block_013_covers

def row11587_large_right_right : List Nat := row11587_block_011 ++ row11587_large_right_right_right
theorem row11587_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11587 row11587_large_right_right :=
  Budget.RowUpperBound.branch row11587_block_011_covers row11587_large_right_right_right_covers

def row11587_large_right : List Nat := row11587_large_right_left ++ row11587_large_right_right
theorem row11587_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11587 row11587_large_right :=
  Budget.RowUpperBound.branch row11587_large_right_left_covers row11587_large_right_right_covers

def row11587_large : List Nat := row11587_large_left ++ row11587_large_right
theorem row11587_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11587 row11587_large :=
  Budget.RowUpperBound.branch row11587_large_left_covers row11587_large_right_covers

def row11587 : List Nat := row11587_large
theorem row11587_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11587 row11587 := row11587_large_covers

end ProjectEulerSolutions.PrimePairLarge
