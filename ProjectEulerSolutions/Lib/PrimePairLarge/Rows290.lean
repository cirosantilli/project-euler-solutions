import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9403_small : List Nat := []
theorem row9403_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9403 row9403_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9403_block_005 : List Nat := [9679, 9697, 10141]
def plan9403_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .possible, .first, .possible, .first, (.firstFactor 19), (.firstFactor 17), .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 7), .second, .first, (.secondFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9403_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9403 row9403_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9403_block_005
  · decide
  · rfl

def row9403_block_006 : List Nat := [10831, 11503, 11941]
def plan9403_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 11), .first, .possible, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .possible, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9403_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9403 row9403_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9403_block_006
  · decide
  · rfl

def row9403_block_007 : List Nat := [12601, 13879, 14197]
def plan9403_block_007 : List Budget.Choice := [(.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, .second, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, (.secondFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row9403_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9403 row9403_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9403_block_007
  · decide
  · rfl

def row9403_block_008 : List Nat := [14653, 15199, 15451, 15619, 16057, 16063, 16339]
def plan9403_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 29), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .second, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9403_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9403 row9403_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9403_block_008
  · decide
  · rfl

def row9403_block_009 : List Nat := [16603]
def plan9403_block_009 : List Budget.Choice := [(.secondFactor 31), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9403_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9403 row9403_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9403_block_009
  · decide
  · rfl

def row9403_block_010 : List Nat := []
theorem row9403_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9403 row9403_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9403_block_011 : List Nat := []
theorem row9403_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9403 row9403_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9403_block_012 : List Nat := []
theorem row9403_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9403 row9403_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9403_block_013 : List Nat := []
theorem row9403_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9403 row9403_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9403_large_left_left : List Nat := row9403_small ++ row9403_block_005
theorem row9403_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9403 row9403_large_left_left :=
  Budget.RowUpperBound.branch row9403_small_covers row9403_block_005_covers

def row9403_large_left_right_right : List Nat := row9403_block_007 ++ row9403_block_008
theorem row9403_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9403 row9403_large_left_right_right :=
  Budget.RowUpperBound.branch row9403_block_007_covers row9403_block_008_covers

def row9403_large_left_right : List Nat := row9403_block_006 ++ row9403_large_left_right_right
theorem row9403_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9403 row9403_large_left_right :=
  Budget.RowUpperBound.branch row9403_block_006_covers row9403_large_left_right_right_covers

def row9403_large_left : List Nat := row9403_large_left_left ++ row9403_large_left_right
theorem row9403_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9403 row9403_large_left :=
  Budget.RowUpperBound.branch row9403_large_left_left_covers row9403_large_left_right_covers

def row9403_large_right_left : List Nat := row9403_block_009 ++ row9403_block_010
theorem row9403_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9403 row9403_large_right_left :=
  Budget.RowUpperBound.branch row9403_block_009_covers row9403_block_010_covers

def row9403_large_right_right_right : List Nat := row9403_block_012 ++ row9403_block_013
theorem row9403_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9403 row9403_large_right_right_right :=
  Budget.RowUpperBound.branch row9403_block_012_covers row9403_block_013_covers

def row9403_large_right_right : List Nat := row9403_block_011 ++ row9403_large_right_right_right
theorem row9403_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9403 row9403_large_right_right :=
  Budget.RowUpperBound.branch row9403_block_011_covers row9403_large_right_right_right_covers

def row9403_large_right : List Nat := row9403_large_right_left ++ row9403_large_right_right
theorem row9403_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9403 row9403_large_right :=
  Budget.RowUpperBound.branch row9403_large_right_left_covers row9403_large_right_right_covers

def row9403_large : List Nat := row9403_large_left ++ row9403_large_right
theorem row9403_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9403 row9403_large :=
  Budget.RowUpperBound.branch row9403_large_left_covers row9403_large_right_covers

def row9403 : List Nat := row9403_large
theorem row9403_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9403 row9403 := row9403_large_covers

def row9413_small : List Nat := []
theorem row9413_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9413 row9413_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9413_block_005 : List Nat := [9677]
def plan9413_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .second, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 17), (.secondFactor 31), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .second, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9413_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9413 row9413_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9413_block_005
  · decide
  · rfl

def row9413_block_006 : List Nat := [10631, 11027, 11447, 11867]
def plan9413_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 31), (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .second, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9413_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9413 row9413_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9413_block_006
  · decide
  · rfl

def row9413_block_007 : List Nat := [12539, 13859]
def plan9413_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 31), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row9413_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9413 row9413_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9413_block_007
  · decide
  · rfl

def row9413_block_008 : List Nat := [15497, 16091]
def plan9413_block_008 : List Budget.Choice := [.first, .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 29), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9413_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9413 row9413_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9413_block_008
  · decide
  · rfl

def row9413_block_009 : List Nat := []
def plan9413_block_009 : List Budget.Choice := [.first, (.secondFactor 23), (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9413_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9413 row9413_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9413_block_009
  · decide
  · rfl

def row9413_block_010 : List Nat := []
theorem row9413_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9413 row9413_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9413_block_011 : List Nat := []
theorem row9413_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9413 row9413_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9413_block_012 : List Nat := []
theorem row9413_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9413 row9413_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9413_block_013 : List Nat := []
theorem row9413_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9413 row9413_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9413_large_left_left : List Nat := row9413_small ++ row9413_block_005
theorem row9413_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9413 row9413_large_left_left :=
  Budget.RowUpperBound.branch row9413_small_covers row9413_block_005_covers

def row9413_large_left_right_right : List Nat := row9413_block_007 ++ row9413_block_008
theorem row9413_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9413 row9413_large_left_right_right :=
  Budget.RowUpperBound.branch row9413_block_007_covers row9413_block_008_covers

def row9413_large_left_right : List Nat := row9413_block_006 ++ row9413_large_left_right_right
theorem row9413_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9413 row9413_large_left_right :=
  Budget.RowUpperBound.branch row9413_block_006_covers row9413_large_left_right_right_covers

def row9413_large_left : List Nat := row9413_large_left_left ++ row9413_large_left_right
theorem row9413_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9413 row9413_large_left :=
  Budget.RowUpperBound.branch row9413_large_left_left_covers row9413_large_left_right_covers

def row9413_large_right_left : List Nat := row9413_block_009 ++ row9413_block_010
theorem row9413_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9413 row9413_large_right_left :=
  Budget.RowUpperBound.branch row9413_block_009_covers row9413_block_010_covers

def row9413_large_right_right_right : List Nat := row9413_block_012 ++ row9413_block_013
theorem row9413_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9413 row9413_large_right_right_right :=
  Budget.RowUpperBound.branch row9413_block_012_covers row9413_block_013_covers

def row9413_large_right_right : List Nat := row9413_block_011 ++ row9413_large_right_right_right
theorem row9413_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9413 row9413_large_right_right :=
  Budget.RowUpperBound.branch row9413_block_011_covers row9413_large_right_right_right_covers

def row9413_large_right : List Nat := row9413_large_right_left ++ row9413_large_right_right
theorem row9413_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9413 row9413_large_right :=
  Budget.RowUpperBound.branch row9413_large_right_left_covers row9413_large_right_right_covers

def row9413_large : List Nat := row9413_large_left ++ row9413_large_right
theorem row9413_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9413 row9413_large :=
  Budget.RowUpperBound.branch row9413_large_left_covers row9413_large_right_covers

def row9413 : List Nat := row9413_large
theorem row9413_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9413 row9413 := row9413_large_covers

def row9419_small : List Nat := []
theorem row9419_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9419 row9419_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9419_block_005 : List Nat := [9719]
def plan9419_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9419_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9419 row9419_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9419_block_005
  · decide
  · rfl

def row9419_block_006 : List Nat := [11933]
def plan9419_block_006 : List Budget.Choice := [(.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .possible, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .second, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9419_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9419 row9419_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9419_block_006
  · decide
  · rfl

def row9419_block_007 : List Nat := [12689, 13121, 14411]
def plan9419_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .possible, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row9419_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9419 row9419_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9419_block_007
  · decide
  · rfl

def row9419_block_008 : List Nat := [14537, 14591, 14621, 15383, 15461, 15773]
def plan9419_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, .possible, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9419_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9419 row9419_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9419_block_008
  · decide
  · rfl

def row9419_block_009 : List Nat := []
def plan9419_block_009 : List Budget.Choice := [.first, .second, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9419_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9419 row9419_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9419_block_009
  · decide
  · rfl

def row9419_block_010 : List Nat := []
theorem row9419_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9419 row9419_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9419_block_011 : List Nat := []
theorem row9419_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9419 row9419_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9419_block_012 : List Nat := []
theorem row9419_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9419 row9419_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9419_block_013 : List Nat := []
theorem row9419_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9419 row9419_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9419_large_left_left : List Nat := row9419_small ++ row9419_block_005
theorem row9419_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9419 row9419_large_left_left :=
  Budget.RowUpperBound.branch row9419_small_covers row9419_block_005_covers

def row9419_large_left_right_right : List Nat := row9419_block_007 ++ row9419_block_008
theorem row9419_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9419 row9419_large_left_right_right :=
  Budget.RowUpperBound.branch row9419_block_007_covers row9419_block_008_covers

def row9419_large_left_right : List Nat := row9419_block_006 ++ row9419_large_left_right_right
theorem row9419_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9419 row9419_large_left_right :=
  Budget.RowUpperBound.branch row9419_block_006_covers row9419_large_left_right_right_covers

def row9419_large_left : List Nat := row9419_large_left_left ++ row9419_large_left_right
theorem row9419_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9419 row9419_large_left :=
  Budget.RowUpperBound.branch row9419_large_left_left_covers row9419_large_left_right_covers

def row9419_large_right_left : List Nat := row9419_block_009 ++ row9419_block_010
theorem row9419_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9419 row9419_large_right_left :=
  Budget.RowUpperBound.branch row9419_block_009_covers row9419_block_010_covers

def row9419_large_right_right_right : List Nat := row9419_block_012 ++ row9419_block_013
theorem row9419_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9419 row9419_large_right_right_right :=
  Budget.RowUpperBound.branch row9419_block_012_covers row9419_block_013_covers

def row9419_large_right_right : List Nat := row9419_block_011 ++ row9419_large_right_right_right
theorem row9419_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9419 row9419_large_right_right :=
  Budget.RowUpperBound.branch row9419_block_011_covers row9419_large_right_right_right_covers

def row9419_large_right : List Nat := row9419_large_right_left ++ row9419_large_right_right
theorem row9419_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9419 row9419_large_right :=
  Budget.RowUpperBound.branch row9419_large_right_left_covers row9419_large_right_right_covers

def row9419_large : List Nat := row9419_large_left ++ row9419_large_right
theorem row9419_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9419 row9419_large :=
  Budget.RowUpperBound.branch row9419_large_left_covers row9419_large_right_covers

def row9419 : List Nat := row9419_large
theorem row9419_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9419 row9419 := row9419_large_covers

def row9421_small : List Nat := []
theorem row9421_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9421 row9421_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9421_block_005 : List Nat := [9679, 9733, 9817, 9973, 10303]
def plan9421_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9421_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9421 row9421_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9421_block_005
  · decide
  · rfl

def row9421_block_006 : List Nat := [10837, 11329, 12043, 12109]
def plan9421_block_006 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .second, .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), .second, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .possible, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9421_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9421 row9421_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9421_block_006
  · decide
  · rfl

def row9421_block_007 : List Nat := [14029]
def plan9421_block_007 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 13), .possible, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row9421_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9421 row9421_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9421_block_007
  · decide
  · rfl

def row9421_block_008 : List Nat := [14827]
def plan9421_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 29), .second, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 31), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 31), (.secondFactor 23), (.firstFactor 11), .first, .first, .first, .second, .first, .second, (.secondFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9421_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9421 row9421_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9421_block_008
  · decide
  · rfl

def row9421_block_009 : List Nat := []
def plan9421_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9421_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9421 row9421_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9421_block_009
  · decide
  · rfl

def row9421_block_010 : List Nat := []
theorem row9421_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9421 row9421_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9421_block_011 : List Nat := []
theorem row9421_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9421 row9421_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9421_block_012 : List Nat := []
theorem row9421_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9421 row9421_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9421_block_013 : List Nat := []
theorem row9421_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9421 row9421_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9421_large_left_left : List Nat := row9421_small ++ row9421_block_005
theorem row9421_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9421 row9421_large_left_left :=
  Budget.RowUpperBound.branch row9421_small_covers row9421_block_005_covers

def row9421_large_left_right_right : List Nat := row9421_block_007 ++ row9421_block_008
theorem row9421_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9421 row9421_large_left_right_right :=
  Budget.RowUpperBound.branch row9421_block_007_covers row9421_block_008_covers

def row9421_large_left_right : List Nat := row9421_block_006 ++ row9421_large_left_right_right
theorem row9421_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9421 row9421_large_left_right :=
  Budget.RowUpperBound.branch row9421_block_006_covers row9421_large_left_right_right_covers

def row9421_large_left : List Nat := row9421_large_left_left ++ row9421_large_left_right
theorem row9421_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9421 row9421_large_left :=
  Budget.RowUpperBound.branch row9421_large_left_left_covers row9421_large_left_right_covers

def row9421_large_right_left : List Nat := row9421_block_009 ++ row9421_block_010
theorem row9421_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9421 row9421_large_right_left :=
  Budget.RowUpperBound.branch row9421_block_009_covers row9421_block_010_covers

def row9421_large_right_right_right : List Nat := row9421_block_012 ++ row9421_block_013
theorem row9421_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9421 row9421_large_right_right_right :=
  Budget.RowUpperBound.branch row9421_block_012_covers row9421_block_013_covers

def row9421_large_right_right : List Nat := row9421_block_011 ++ row9421_large_right_right_right
theorem row9421_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9421 row9421_large_right_right :=
  Budget.RowUpperBound.branch row9421_block_011_covers row9421_large_right_right_right_covers

def row9421_large_right : List Nat := row9421_large_right_left ++ row9421_large_right_right
theorem row9421_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9421 row9421_large_right :=
  Budget.RowUpperBound.branch row9421_large_right_left_covers row9421_large_right_right_covers

def row9421_large : List Nat := row9421_large_left ++ row9421_large_right
theorem row9421_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9421 row9421_large :=
  Budget.RowUpperBound.branch row9421_large_left_covers row9421_large_right_covers

def row9421 : List Nat := row9421_large
theorem row9421_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9421 row9421 := row9421_large_covers

end ProjectEulerSolutions.PrimePairLarge
