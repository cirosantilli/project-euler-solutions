import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10957_small : List Nat := []
theorem row10957_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10957 row10957_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10957_block_005 : List Nat := []
theorem row10957_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10957 row10957_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10957_block_006 : List Nat := []
def plan10957_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .second, .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 31), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row10957_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10957 row10957_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10957_block_006
  · decide
  · rfl

def row10957_block_007 : List Nat := [12613, 13183, 14341]
def plan10957_block_007 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .possible, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .second, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row10957_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10957 row10957_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10957_block_007
  · decide
  · rfl

def row10957_block_008 : List Nat := [14563]
def plan10957_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10957_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10957 row10957_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10957_block_008
  · decide
  · rfl

def row10957_block_009 : List Nat := []
theorem row10957_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10957 row10957_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10957_block_010 : List Nat := []
theorem row10957_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10957 row10957_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10957_block_011 : List Nat := []
theorem row10957_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10957 row10957_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10957_block_012 : List Nat := []
theorem row10957_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10957 row10957_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10957_block_013 : List Nat := []
theorem row10957_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10957 row10957_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10957_large_left_left : List Nat := row10957_small ++ row10957_block_005
theorem row10957_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10957 row10957_large_left_left :=
  Budget.RowUpperBound.branch row10957_small_covers row10957_block_005_covers

def row10957_large_left_right_right : List Nat := row10957_block_007 ++ row10957_block_008
theorem row10957_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10957 row10957_large_left_right_right :=
  Budget.RowUpperBound.branch row10957_block_007_covers row10957_block_008_covers

def row10957_large_left_right : List Nat := row10957_block_006 ++ row10957_large_left_right_right
theorem row10957_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10957 row10957_large_left_right :=
  Budget.RowUpperBound.branch row10957_block_006_covers row10957_large_left_right_right_covers

def row10957_large_left : List Nat := row10957_large_left_left ++ row10957_large_left_right
theorem row10957_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10957 row10957_large_left :=
  Budget.RowUpperBound.branch row10957_large_left_left_covers row10957_large_left_right_covers

def row10957_large_right_left : List Nat := row10957_block_009 ++ row10957_block_010
theorem row10957_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10957 row10957_large_right_left :=
  Budget.RowUpperBound.branch row10957_block_009_covers row10957_block_010_covers

def row10957_large_right_right_right : List Nat := row10957_block_012 ++ row10957_block_013
theorem row10957_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10957 row10957_large_right_right_right :=
  Budget.RowUpperBound.branch row10957_block_012_covers row10957_block_013_covers

def row10957_large_right_right : List Nat := row10957_block_011 ++ row10957_large_right_right_right
theorem row10957_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10957 row10957_large_right_right :=
  Budget.RowUpperBound.branch row10957_block_011_covers row10957_large_right_right_right_covers

def row10957_large_right : List Nat := row10957_large_right_left ++ row10957_large_right_right
theorem row10957_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10957 row10957_large_right :=
  Budget.RowUpperBound.branch row10957_large_right_left_covers row10957_large_right_right_covers

def row10957_large : List Nat := row10957_large_left ++ row10957_large_right
theorem row10957_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10957 row10957_large :=
  Budget.RowUpperBound.branch row10957_large_left_covers row10957_large_right_covers

def row10957 : List Nat := row10957_large
theorem row10957_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10957 row10957 := row10957_large_covers

def row10973_small : List Nat := []
theorem row10973_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10973 row10973_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10973_block_005 : List Nat := []
theorem row10973_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10973 row10973_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10973_block_006 : List Nat := [11939, 12143]
def plan10973_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 17), .second, .possible, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .second, (.secondFactor 7), .possible, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row10973_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10973 row10973_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10973_block_006
  · decide
  · rfl

def row10973_block_007 : List Nat := [12569, 12659, 13577, 13781, 13901]
def plan10973_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 31), .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .second, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10973_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10973 row10973_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10973_block_007
  · decide
  · rfl

def row10973_block_008 : List Nat := [14879]
def plan10973_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 37), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10973_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10973 row10973_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10973_block_008
  · decide
  · rfl

def row10973_block_009 : List Nat := []
theorem row10973_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10973 row10973_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10973_block_010 : List Nat := []
theorem row10973_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10973 row10973_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10973_block_011 : List Nat := []
theorem row10973_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10973 row10973_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10973_block_012 : List Nat := []
theorem row10973_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10973 row10973_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10973_block_013 : List Nat := []
theorem row10973_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10973 row10973_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10973_large_left_left : List Nat := row10973_small ++ row10973_block_005
theorem row10973_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10973 row10973_large_left_left :=
  Budget.RowUpperBound.branch row10973_small_covers row10973_block_005_covers

def row10973_large_left_right_right : List Nat := row10973_block_007 ++ row10973_block_008
theorem row10973_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10973 row10973_large_left_right_right :=
  Budget.RowUpperBound.branch row10973_block_007_covers row10973_block_008_covers

def row10973_large_left_right : List Nat := row10973_block_006 ++ row10973_large_left_right_right
theorem row10973_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10973 row10973_large_left_right :=
  Budget.RowUpperBound.branch row10973_block_006_covers row10973_large_left_right_right_covers

def row10973_large_left : List Nat := row10973_large_left_left ++ row10973_large_left_right
theorem row10973_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10973 row10973_large_left :=
  Budget.RowUpperBound.branch row10973_large_left_left_covers row10973_large_left_right_covers

def row10973_large_right_left : List Nat := row10973_block_009 ++ row10973_block_010
theorem row10973_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10973 row10973_large_right_left :=
  Budget.RowUpperBound.branch row10973_block_009_covers row10973_block_010_covers

def row10973_large_right_right_right : List Nat := row10973_block_012 ++ row10973_block_013
theorem row10973_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10973 row10973_large_right_right_right :=
  Budget.RowUpperBound.branch row10973_block_012_covers row10973_block_013_covers

def row10973_large_right_right : List Nat := row10973_block_011 ++ row10973_large_right_right_right
theorem row10973_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10973 row10973_large_right_right :=
  Budget.RowUpperBound.branch row10973_block_011_covers row10973_large_right_right_right_covers

def row10973_large_right : List Nat := row10973_large_right_left ++ row10973_large_right_right
theorem row10973_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10973 row10973_large_right :=
  Budget.RowUpperBound.branch row10973_large_right_left_covers row10973_large_right_right_covers

def row10973_large : List Nat := row10973_large_left ++ row10973_large_right
theorem row10973_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10973 row10973_large :=
  Budget.RowUpperBound.branch row10973_large_left_covers row10973_large_right_covers

def row10973 : List Nat := row10973_large
theorem row10973_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10973 row10973 := row10973_large_covers

def row10979_small : List Nat := []
theorem row10979_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10979 row10979_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10979_block_005 : List Nat := []
theorem row10979_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10979 row10979_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10979_block_006 : List Nat := [11621, 11987, 12401]
def plan10979_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .second, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10979_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10979 row10979_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10979_block_006
  · decide
  · rfl

def row10979_block_007 : List Nat := [12671, 13259]
def plan10979_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), (.firstFactor 23), .possible, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 13), (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .first, .second, .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10979_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10979 row10979_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10979_block_007
  · decide
  · rfl

def row10979_block_008 : List Nat := [14771, 14843, 15053]
def plan10979_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 17), .second, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .second, .first, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10979_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10979 row10979_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10979_block_008
  · decide
  · rfl

def row10979_block_009 : List Nat := []
theorem row10979_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10979 row10979_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10979_block_010 : List Nat := []
theorem row10979_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10979 row10979_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10979_block_011 : List Nat := []
theorem row10979_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10979 row10979_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10979_block_012 : List Nat := []
theorem row10979_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10979 row10979_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10979_block_013 : List Nat := []
theorem row10979_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10979 row10979_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10979_large_left_left : List Nat := row10979_small ++ row10979_block_005
theorem row10979_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10979 row10979_large_left_left :=
  Budget.RowUpperBound.branch row10979_small_covers row10979_block_005_covers

def row10979_large_left_right_right : List Nat := row10979_block_007 ++ row10979_block_008
theorem row10979_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10979 row10979_large_left_right_right :=
  Budget.RowUpperBound.branch row10979_block_007_covers row10979_block_008_covers

def row10979_large_left_right : List Nat := row10979_block_006 ++ row10979_large_left_right_right
theorem row10979_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10979 row10979_large_left_right :=
  Budget.RowUpperBound.branch row10979_block_006_covers row10979_large_left_right_right_covers

def row10979_large_left : List Nat := row10979_large_left_left ++ row10979_large_left_right
theorem row10979_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10979 row10979_large_left :=
  Budget.RowUpperBound.branch row10979_large_left_left_covers row10979_large_left_right_covers

def row10979_large_right_left : List Nat := row10979_block_009 ++ row10979_block_010
theorem row10979_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10979 row10979_large_right_left :=
  Budget.RowUpperBound.branch row10979_block_009_covers row10979_block_010_covers

def row10979_large_right_right_right : List Nat := row10979_block_012 ++ row10979_block_013
theorem row10979_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10979 row10979_large_right_right_right :=
  Budget.RowUpperBound.branch row10979_block_012_covers row10979_block_013_covers

def row10979_large_right_right : List Nat := row10979_block_011 ++ row10979_large_right_right_right
theorem row10979_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10979 row10979_large_right_right :=
  Budget.RowUpperBound.branch row10979_block_011_covers row10979_large_right_right_right_covers

def row10979_large_right : List Nat := row10979_large_right_left ++ row10979_large_right_right
theorem row10979_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10979 row10979_large_right :=
  Budget.RowUpperBound.branch row10979_large_right_left_covers row10979_large_right_right_covers

def row10979_large : List Nat := row10979_large_left ++ row10979_large_right
theorem row10979_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10979 row10979_large :=
  Budget.RowUpperBound.branch row10979_large_left_covers row10979_large_right_covers

def row10979 : List Nat := row10979_large
theorem row10979_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10979 row10979 := row10979_large_covers

def row10987_small : List Nat := []
theorem row10987_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10987 row10987_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10987_block_005 : List Nat := []
theorem row10987_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10987 row10987_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10987_block_006 : List Nat := [11491, 11701, 11743, 12379, 12409]
def plan10987_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row10987_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10987 row10987_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10987_block_006
  · decide
  · rfl

def row10987_block_007 : List Nat := [12613, 13003, 13999]
def plan10987_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .possible, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 19), .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10987_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10987 row10987_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10987_block_007
  · decide
  · rfl

def row10987_block_008 : List Nat := []
def plan10987_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10987_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10987 row10987_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10987_block_008
  · decide
  · rfl

def row10987_block_009 : List Nat := []
theorem row10987_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10987 row10987_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10987_block_010 : List Nat := []
theorem row10987_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10987 row10987_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10987_block_011 : List Nat := []
theorem row10987_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10987 row10987_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10987_block_012 : List Nat := []
theorem row10987_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10987 row10987_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10987_block_013 : List Nat := []
theorem row10987_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10987 row10987_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10987_large_left_left : List Nat := row10987_small ++ row10987_block_005
theorem row10987_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10987 row10987_large_left_left :=
  Budget.RowUpperBound.branch row10987_small_covers row10987_block_005_covers

def row10987_large_left_right_right : List Nat := row10987_block_007 ++ row10987_block_008
theorem row10987_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10987 row10987_large_left_right_right :=
  Budget.RowUpperBound.branch row10987_block_007_covers row10987_block_008_covers

def row10987_large_left_right : List Nat := row10987_block_006 ++ row10987_large_left_right_right
theorem row10987_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10987 row10987_large_left_right :=
  Budget.RowUpperBound.branch row10987_block_006_covers row10987_large_left_right_right_covers

def row10987_large_left : List Nat := row10987_large_left_left ++ row10987_large_left_right
theorem row10987_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10987 row10987_large_left :=
  Budget.RowUpperBound.branch row10987_large_left_left_covers row10987_large_left_right_covers

def row10987_large_right_left : List Nat := row10987_block_009 ++ row10987_block_010
theorem row10987_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10987 row10987_large_right_left :=
  Budget.RowUpperBound.branch row10987_block_009_covers row10987_block_010_covers

def row10987_large_right_right_right : List Nat := row10987_block_012 ++ row10987_block_013
theorem row10987_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10987 row10987_large_right_right_right :=
  Budget.RowUpperBound.branch row10987_block_012_covers row10987_block_013_covers

def row10987_large_right_right : List Nat := row10987_block_011 ++ row10987_large_right_right_right
theorem row10987_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10987 row10987_large_right_right :=
  Budget.RowUpperBound.branch row10987_block_011_covers row10987_large_right_right_right_covers

def row10987_large_right : List Nat := row10987_large_right_left ++ row10987_large_right_right
theorem row10987_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10987 row10987_large_right :=
  Budget.RowUpperBound.branch row10987_large_right_left_covers row10987_large_right_right_covers

def row10987_large : List Nat := row10987_large_left ++ row10987_large_right
theorem row10987_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10987 row10987_large :=
  Budget.RowUpperBound.branch row10987_large_left_covers row10987_large_right_covers

def row10987 : List Nat := row10987_large
theorem row10987_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10987 row10987 := row10987_large_covers

end ProjectEulerSolutions.PrimePairLarge
