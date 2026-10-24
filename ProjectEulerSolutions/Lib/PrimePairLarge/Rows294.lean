import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9521_small : List Nat := []
theorem row9521_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9521 row9521_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9521_block_005 : List Nat := [9767, 10091]
def plan9521_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9521_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9521 row9521_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9521_block_005
  · decide
  · rfl

def row9521_block_006 : List Nat := [10427, 10613, 10949, 11351]
def plan9521_block_006 : List Budget.Choice := [(.secondFactor 17), .first, .possible, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .second, .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .second, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 29), (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9521_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9521 row9521_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9521_block_006
  · decide
  · rfl

def row9521_block_007 : List Nat := [13187, 13691]
def plan9521_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 11), .first, .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9521_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9521 row9521_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9521_block_007
  · decide
  · rfl

def row9521_block_008 : List Nat := [14519, 14753, 14939]
def plan9521_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 11), .second, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .possible, .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9521_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9521 row9521_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9521_block_008
  · decide
  · rfl

def row9521_block_009 : List Nat := []
theorem row9521_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9521 row9521_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9521_block_010 : List Nat := []
theorem row9521_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9521 row9521_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9521_block_011 : List Nat := []
theorem row9521_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9521 row9521_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9521_block_012 : List Nat := []
theorem row9521_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9521 row9521_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9521_block_013 : List Nat := []
theorem row9521_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9521 row9521_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9521_large_left_left : List Nat := row9521_small ++ row9521_block_005
theorem row9521_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9521 row9521_large_left_left :=
  Budget.RowUpperBound.branch row9521_small_covers row9521_block_005_covers

def row9521_large_left_right_right : List Nat := row9521_block_007 ++ row9521_block_008
theorem row9521_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9521 row9521_large_left_right_right :=
  Budget.RowUpperBound.branch row9521_block_007_covers row9521_block_008_covers

def row9521_large_left_right : List Nat := row9521_block_006 ++ row9521_large_left_right_right
theorem row9521_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9521 row9521_large_left_right :=
  Budget.RowUpperBound.branch row9521_block_006_covers row9521_large_left_right_right_covers

def row9521_large_left : List Nat := row9521_large_left_left ++ row9521_large_left_right
theorem row9521_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9521 row9521_large_left :=
  Budget.RowUpperBound.branch row9521_large_left_left_covers row9521_large_left_right_covers

def row9521_large_right_left : List Nat := row9521_block_009 ++ row9521_block_010
theorem row9521_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9521 row9521_large_right_left :=
  Budget.RowUpperBound.branch row9521_block_009_covers row9521_block_010_covers

def row9521_large_right_right_right : List Nat := row9521_block_012 ++ row9521_block_013
theorem row9521_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9521 row9521_large_right_right_right :=
  Budget.RowUpperBound.branch row9521_block_012_covers row9521_block_013_covers

def row9521_large_right_right : List Nat := row9521_block_011 ++ row9521_large_right_right_right
theorem row9521_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9521 row9521_large_right_right :=
  Budget.RowUpperBound.branch row9521_block_011_covers row9521_large_right_right_right_covers

def row9521_large_right : List Nat := row9521_large_right_left ++ row9521_large_right_right
theorem row9521_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9521 row9521_large_right :=
  Budget.RowUpperBound.branch row9521_large_right_left_covers row9521_large_right_right_covers

def row9521_large : List Nat := row9521_large_left ++ row9521_large_right
theorem row9521_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9521 row9521_large :=
  Budget.RowUpperBound.branch row9521_large_left_covers row9521_large_right_covers

def row9521 : List Nat := row9521_large
theorem row9521_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9521 row9521 := row9521_large_covers

def row9533_small : List Nat := []
theorem row9533_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9533 row9533_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9533_block_005 : List Nat := [10079]
def plan9533_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 23), .first, .possible, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9533_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9533 row9533_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9533_block_005
  · decide
  · rfl

def row9533_block_006 : List Nat := [11399, 11621, 12323]
def plan9533_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9533_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9533 row9533_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9533_block_006
  · decide
  · rfl

def row9533_block_007 : List Nat := []
def plan9533_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9533_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9533 row9533_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9533_block_007
  · decide
  · rfl

def row9533_block_008 : List Nat := [15263, 15497, 16187]
def plan9533_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .second, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .second, .first, (.firstFactor 19), (.secondFactor 23), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9533_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9533 row9533_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9533_block_008
  · decide
  · rfl

def row9533_block_009 : List Nat := []
theorem row9533_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9533 row9533_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9533_block_010 : List Nat := []
theorem row9533_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9533 row9533_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9533_block_011 : List Nat := []
theorem row9533_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9533 row9533_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9533_block_012 : List Nat := []
theorem row9533_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9533 row9533_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9533_block_013 : List Nat := []
theorem row9533_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9533 row9533_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9533_large_left_left : List Nat := row9533_small ++ row9533_block_005
theorem row9533_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9533 row9533_large_left_left :=
  Budget.RowUpperBound.branch row9533_small_covers row9533_block_005_covers

def row9533_large_left_right_right : List Nat := row9533_block_007 ++ row9533_block_008
theorem row9533_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9533 row9533_large_left_right_right :=
  Budget.RowUpperBound.branch row9533_block_007_covers row9533_block_008_covers

def row9533_large_left_right : List Nat := row9533_block_006 ++ row9533_large_left_right_right
theorem row9533_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9533 row9533_large_left_right :=
  Budget.RowUpperBound.branch row9533_block_006_covers row9533_large_left_right_right_covers

def row9533_large_left : List Nat := row9533_large_left_left ++ row9533_large_left_right
theorem row9533_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9533 row9533_large_left :=
  Budget.RowUpperBound.branch row9533_large_left_left_covers row9533_large_left_right_covers

def row9533_large_right_left : List Nat := row9533_block_009 ++ row9533_block_010
theorem row9533_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9533 row9533_large_right_left :=
  Budget.RowUpperBound.branch row9533_block_009_covers row9533_block_010_covers

def row9533_large_right_right_right : List Nat := row9533_block_012 ++ row9533_block_013
theorem row9533_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9533 row9533_large_right_right_right :=
  Budget.RowUpperBound.branch row9533_block_012_covers row9533_block_013_covers

def row9533_large_right_right : List Nat := row9533_block_011 ++ row9533_large_right_right_right
theorem row9533_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9533 row9533_large_right_right :=
  Budget.RowUpperBound.branch row9533_block_011_covers row9533_large_right_right_right_covers

def row9533_large_right : List Nat := row9533_large_right_left ++ row9533_large_right_right
theorem row9533_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9533 row9533_large_right :=
  Budget.RowUpperBound.branch row9533_large_right_left_covers row9533_large_right_right_covers

def row9533_large : List Nat := row9533_large_left ++ row9533_large_right
theorem row9533_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9533 row9533_large :=
  Budget.RowUpperBound.branch row9533_large_left_covers row9533_large_right_covers

def row9533 : List Nat := row9533_large
theorem row9533_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9533 row9533 := row9533_large_covers

def row9539_small : List Nat := []
theorem row9539_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9539 row9539_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9539_block_005 : List Nat := []
def plan9539_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9539_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9539 row9539_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9539_block_005
  · decide
  · rfl

def row9539_block_006 : List Nat := [10733, 11159, 12269]
def plan9539_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .second, .first, .first, .first, .possible, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, .first, .possible, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 37), (.firstFactor 7), .second, .second, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9539_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9539 row9539_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9539_block_006
  · decide
  · rfl

def row9539_block_007 : List Nat := [12899, 13007]
def plan9539_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, .first, (.firstFactor 7), .second, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 31), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9539_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9539 row9539_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9539_block_007
  · decide
  · rfl

def row9539_block_008 : List Nat := [15527, 15737]
def plan9539_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .possible, .first, .second, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 11), .second]
set_option maxRecDepth 2048 in
theorem row9539_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9539 row9539_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9539_block_008
  · decide
  · rfl

def row9539_block_009 : List Nat := []
theorem row9539_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9539 row9539_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9539_block_010 : List Nat := []
theorem row9539_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9539 row9539_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9539_block_011 : List Nat := []
theorem row9539_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9539 row9539_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9539_block_012 : List Nat := []
theorem row9539_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9539 row9539_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9539_block_013 : List Nat := []
theorem row9539_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9539 row9539_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9539_large_left_left : List Nat := row9539_small ++ row9539_block_005
theorem row9539_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9539 row9539_large_left_left :=
  Budget.RowUpperBound.branch row9539_small_covers row9539_block_005_covers

def row9539_large_left_right_right : List Nat := row9539_block_007 ++ row9539_block_008
theorem row9539_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9539 row9539_large_left_right_right :=
  Budget.RowUpperBound.branch row9539_block_007_covers row9539_block_008_covers

def row9539_large_left_right : List Nat := row9539_block_006 ++ row9539_large_left_right_right
theorem row9539_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9539 row9539_large_left_right :=
  Budget.RowUpperBound.branch row9539_block_006_covers row9539_large_left_right_right_covers

def row9539_large_left : List Nat := row9539_large_left_left ++ row9539_large_left_right
theorem row9539_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9539 row9539_large_left :=
  Budget.RowUpperBound.branch row9539_large_left_left_covers row9539_large_left_right_covers

def row9539_large_right_left : List Nat := row9539_block_009 ++ row9539_block_010
theorem row9539_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9539 row9539_large_right_left :=
  Budget.RowUpperBound.branch row9539_block_009_covers row9539_block_010_covers

def row9539_large_right_right_right : List Nat := row9539_block_012 ++ row9539_block_013
theorem row9539_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9539 row9539_large_right_right_right :=
  Budget.RowUpperBound.branch row9539_block_012_covers row9539_block_013_covers

def row9539_large_right_right : List Nat := row9539_block_011 ++ row9539_large_right_right_right
theorem row9539_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9539 row9539_large_right_right :=
  Budget.RowUpperBound.branch row9539_block_011_covers row9539_large_right_right_right_covers

def row9539_large_right : List Nat := row9539_large_right_left ++ row9539_large_right_right
theorem row9539_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9539 row9539_large_right :=
  Budget.RowUpperBound.branch row9539_large_right_left_covers row9539_large_right_right_covers

def row9539_large : List Nat := row9539_large_left ++ row9539_large_right
theorem row9539_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9539 row9539_large :=
  Budget.RowUpperBound.branch row9539_large_left_covers row9539_large_right_covers

def row9539 : List Nat := row9539_large
theorem row9539_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9539 row9539 := row9539_large_covers

def row9547_small : List Nat := []
theorem row9547_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9547 row9547_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9547_block_005 : List Nat := [10333, 10357]
def plan9547_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .second, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .possible, .first, .first, .possible, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9547_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9547 row9547_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9547_block_005
  · decide
  · rfl

def row9547_block_006 : List Nat := [12391]
def plan9547_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 11), (.secondFactor 17), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .possible, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9547_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9547 row9547_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9547_block_006
  · decide
  · rfl

def row9547_block_007 : List Nat := [13297, 13381, 13723, 13873, 14143]
def plan9547_block_007 : List Budget.Choice := [(.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .second, (.firstFactor 7), (.secondFactor 17), (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .second]
set_option maxRecDepth 2048 in
theorem row9547_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9547 row9547_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9547_block_007
  · decide
  · rfl

def row9547_block_008 : List Nat := [15991]
def plan9547_block_008 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 31), (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .second, (.firstFactor 23), (.secondFactor 11), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .possible, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 11), (.secondFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), .first, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9547_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9547 row9547_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9547_block_008
  · decide
  · rfl

def row9547_block_009 : List Nat := []
theorem row9547_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9547 row9547_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9547_block_010 : List Nat := []
theorem row9547_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9547 row9547_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9547_block_011 : List Nat := []
theorem row9547_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9547 row9547_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9547_block_012 : List Nat := []
theorem row9547_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9547 row9547_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9547_block_013 : List Nat := []
theorem row9547_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9547 row9547_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9547_large_left_left : List Nat := row9547_small ++ row9547_block_005
theorem row9547_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9547 row9547_large_left_left :=
  Budget.RowUpperBound.branch row9547_small_covers row9547_block_005_covers

def row9547_large_left_right_right : List Nat := row9547_block_007 ++ row9547_block_008
theorem row9547_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9547 row9547_large_left_right_right :=
  Budget.RowUpperBound.branch row9547_block_007_covers row9547_block_008_covers

def row9547_large_left_right : List Nat := row9547_block_006 ++ row9547_large_left_right_right
theorem row9547_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9547 row9547_large_left_right :=
  Budget.RowUpperBound.branch row9547_block_006_covers row9547_large_left_right_right_covers

def row9547_large_left : List Nat := row9547_large_left_left ++ row9547_large_left_right
theorem row9547_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9547 row9547_large_left :=
  Budget.RowUpperBound.branch row9547_large_left_left_covers row9547_large_left_right_covers

def row9547_large_right_left : List Nat := row9547_block_009 ++ row9547_block_010
theorem row9547_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9547 row9547_large_right_left :=
  Budget.RowUpperBound.branch row9547_block_009_covers row9547_block_010_covers

def row9547_large_right_right_right : List Nat := row9547_block_012 ++ row9547_block_013
theorem row9547_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9547 row9547_large_right_right_right :=
  Budget.RowUpperBound.branch row9547_block_012_covers row9547_block_013_covers

def row9547_large_right_right : List Nat := row9547_block_011 ++ row9547_large_right_right_right
theorem row9547_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9547 row9547_large_right_right :=
  Budget.RowUpperBound.branch row9547_block_011_covers row9547_large_right_right_right_covers

def row9547_large_right : List Nat := row9547_large_right_left ++ row9547_large_right_right
theorem row9547_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9547 row9547_large_right :=
  Budget.RowUpperBound.branch row9547_large_right_left_covers row9547_large_right_right_covers

def row9547_large : List Nat := row9547_large_left ++ row9547_large_right
theorem row9547_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9547 row9547_large :=
  Budget.RowUpperBound.branch row9547_large_left_covers row9547_large_right_covers

def row9547 : List Nat := row9547_large
theorem row9547_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9547 row9547 := row9547_large_covers

end ProjectEulerSolutions.PrimePairLarge
