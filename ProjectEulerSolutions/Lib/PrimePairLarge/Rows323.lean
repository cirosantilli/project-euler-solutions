import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10613_small : List Nat := []
theorem row10613_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10613 row10613_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10613_block_005 : List Nat := []
theorem row10613_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10613 row10613_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10613_block_006 : List Nat := [11807]
def plan10613_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 29), .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 11), .possible, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10613_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10613 row10613_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10613_block_006
  · decide
  · rfl

def row10613_block_007 : List Nat := [12809, 13397]
def plan10613_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .second, .first, .second, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .second, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 37), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10613_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10613 row10613_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10613_block_007
  · decide
  · rfl

def row10613_block_008 : List Nat := [14699, 14747]
def plan10613_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10613_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10613 row10613_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10613_block_008
  · decide
  · rfl

def row10613_block_009 : List Nat := []
theorem row10613_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10613 row10613_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10613_block_010 : List Nat := []
theorem row10613_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10613 row10613_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10613_block_011 : List Nat := []
theorem row10613_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10613 row10613_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10613_block_012 : List Nat := []
theorem row10613_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10613 row10613_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10613_block_013 : List Nat := []
theorem row10613_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10613 row10613_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10613_large_left_left : List Nat := row10613_small ++ row10613_block_005
theorem row10613_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10613 row10613_large_left_left :=
  Budget.RowUpperBound.branch row10613_small_covers row10613_block_005_covers

def row10613_large_left_right_right : List Nat := row10613_block_007 ++ row10613_block_008
theorem row10613_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10613 row10613_large_left_right_right :=
  Budget.RowUpperBound.branch row10613_block_007_covers row10613_block_008_covers

def row10613_large_left_right : List Nat := row10613_block_006 ++ row10613_large_left_right_right
theorem row10613_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10613 row10613_large_left_right :=
  Budget.RowUpperBound.branch row10613_block_006_covers row10613_large_left_right_right_covers

def row10613_large_left : List Nat := row10613_large_left_left ++ row10613_large_left_right
theorem row10613_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10613 row10613_large_left :=
  Budget.RowUpperBound.branch row10613_large_left_left_covers row10613_large_left_right_covers

def row10613_large_right_left : List Nat := row10613_block_009 ++ row10613_block_010
theorem row10613_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10613 row10613_large_right_left :=
  Budget.RowUpperBound.branch row10613_block_009_covers row10613_block_010_covers

def row10613_large_right_right_right : List Nat := row10613_block_012 ++ row10613_block_013
theorem row10613_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10613 row10613_large_right_right_right :=
  Budget.RowUpperBound.branch row10613_block_012_covers row10613_block_013_covers

def row10613_large_right_right : List Nat := row10613_block_011 ++ row10613_large_right_right_right
theorem row10613_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10613 row10613_large_right_right :=
  Budget.RowUpperBound.branch row10613_block_011_covers row10613_large_right_right_right_covers

def row10613_large_right : List Nat := row10613_large_right_left ++ row10613_large_right_right
theorem row10613_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10613 row10613_large_right :=
  Budget.RowUpperBound.branch row10613_large_right_left_covers row10613_large_right_right_covers

def row10613_large : List Nat := row10613_large_left ++ row10613_large_right
theorem row10613_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10613 row10613_large :=
  Budget.RowUpperBound.branch row10613_large_left_covers row10613_large_right_covers

def row10613 : List Nat := row10613_large
theorem row10613_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10613 row10613 := row10613_large_covers

def row10627_small : List Nat := []
theorem row10627_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10627 row10627_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10627_block_005 : List Nat := []
theorem row10627_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10627 row10627_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10627_block_006 : List Nat := []
def plan10627_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 29), (.secondFactor 37), .first, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 37), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row10627_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10627 row10627_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10627_block_006
  · decide
  · rfl

def row10627_block_007 : List Nat := [13009, 13627, 13999, 14143, 14281]
def plan10627_block_007 : List Budget.Choice := [(.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 37), .first, .first, .possible, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row10627_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10627 row10627_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10627_block_007
  · decide
  · rfl

def row10627_block_008 : List Nat := [14923, 15193]
def plan10627_block_008 : List Budget.Choice := [(.firstFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10627_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10627 row10627_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10627_block_008
  · decide
  · rfl

def row10627_block_009 : List Nat := []
theorem row10627_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10627 row10627_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10627_block_010 : List Nat := []
theorem row10627_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10627 row10627_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10627_block_011 : List Nat := []
theorem row10627_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10627 row10627_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10627_block_012 : List Nat := []
theorem row10627_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10627 row10627_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10627_block_013 : List Nat := []
theorem row10627_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10627 row10627_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10627_large_left_left : List Nat := row10627_small ++ row10627_block_005
theorem row10627_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10627 row10627_large_left_left :=
  Budget.RowUpperBound.branch row10627_small_covers row10627_block_005_covers

def row10627_large_left_right_right : List Nat := row10627_block_007 ++ row10627_block_008
theorem row10627_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10627 row10627_large_left_right_right :=
  Budget.RowUpperBound.branch row10627_block_007_covers row10627_block_008_covers

def row10627_large_left_right : List Nat := row10627_block_006 ++ row10627_large_left_right_right
theorem row10627_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10627 row10627_large_left_right :=
  Budget.RowUpperBound.branch row10627_block_006_covers row10627_large_left_right_right_covers

def row10627_large_left : List Nat := row10627_large_left_left ++ row10627_large_left_right
theorem row10627_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10627 row10627_large_left :=
  Budget.RowUpperBound.branch row10627_large_left_left_covers row10627_large_left_right_covers

def row10627_large_right_left : List Nat := row10627_block_009 ++ row10627_block_010
theorem row10627_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10627 row10627_large_right_left :=
  Budget.RowUpperBound.branch row10627_block_009_covers row10627_block_010_covers

def row10627_large_right_right_right : List Nat := row10627_block_012 ++ row10627_block_013
theorem row10627_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10627 row10627_large_right_right_right :=
  Budget.RowUpperBound.branch row10627_block_012_covers row10627_block_013_covers

def row10627_large_right_right : List Nat := row10627_block_011 ++ row10627_large_right_right_right
theorem row10627_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10627 row10627_large_right_right :=
  Budget.RowUpperBound.branch row10627_block_011_covers row10627_large_right_right_right_covers

def row10627_large_right : List Nat := row10627_large_right_left ++ row10627_large_right_right
theorem row10627_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10627 row10627_large_right :=
  Budget.RowUpperBound.branch row10627_large_right_left_covers row10627_large_right_right_covers

def row10627_large : List Nat := row10627_large_left ++ row10627_large_right
theorem row10627_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10627 row10627_large :=
  Budget.RowUpperBound.branch row10627_large_left_covers row10627_large_right_covers

def row10627 : List Nat := row10627_large
theorem row10627_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10627 row10627 := row10627_large_covers

def row10631_small : List Nat := []
theorem row10631_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10631 row10631_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10631_block_005 : List Nat := []
theorem row10631_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10631 row10631_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10631_block_006 : List Nat := [10847, 11069]
def plan10631_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10631_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10631 row10631_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10631_block_006
  · decide
  · rfl

def row10631_block_007 : List Nat := [14207]
def plan10631_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 23), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 37), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, .possible, .first, .first, .second, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10631_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10631 row10631_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10631_block_007
  · decide
  · rfl

def row10631_block_008 : List Nat := [14561]
def plan10631_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 19), .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .second, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 37), .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 37), .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10631_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10631 row10631_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10631_block_008
  · decide
  · rfl

def row10631_block_009 : List Nat := []
theorem row10631_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10631 row10631_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10631_block_010 : List Nat := []
theorem row10631_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10631 row10631_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10631_block_011 : List Nat := []
theorem row10631_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10631 row10631_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10631_block_012 : List Nat := []
theorem row10631_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10631 row10631_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10631_block_013 : List Nat := []
theorem row10631_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10631 row10631_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10631_large_left_left : List Nat := row10631_small ++ row10631_block_005
theorem row10631_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10631 row10631_large_left_left :=
  Budget.RowUpperBound.branch row10631_small_covers row10631_block_005_covers

def row10631_large_left_right_right : List Nat := row10631_block_007 ++ row10631_block_008
theorem row10631_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10631 row10631_large_left_right_right :=
  Budget.RowUpperBound.branch row10631_block_007_covers row10631_block_008_covers

def row10631_large_left_right : List Nat := row10631_block_006 ++ row10631_large_left_right_right
theorem row10631_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10631 row10631_large_left_right :=
  Budget.RowUpperBound.branch row10631_block_006_covers row10631_large_left_right_right_covers

def row10631_large_left : List Nat := row10631_large_left_left ++ row10631_large_left_right
theorem row10631_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10631 row10631_large_left :=
  Budget.RowUpperBound.branch row10631_large_left_left_covers row10631_large_left_right_covers

def row10631_large_right_left : List Nat := row10631_block_009 ++ row10631_block_010
theorem row10631_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10631 row10631_large_right_left :=
  Budget.RowUpperBound.branch row10631_block_009_covers row10631_block_010_covers

def row10631_large_right_right_right : List Nat := row10631_block_012 ++ row10631_block_013
theorem row10631_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10631 row10631_large_right_right_right :=
  Budget.RowUpperBound.branch row10631_block_012_covers row10631_block_013_covers

def row10631_large_right_right : List Nat := row10631_block_011 ++ row10631_large_right_right_right
theorem row10631_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10631 row10631_large_right_right :=
  Budget.RowUpperBound.branch row10631_block_011_covers row10631_large_right_right_right_covers

def row10631_large_right : List Nat := row10631_large_right_left ++ row10631_large_right_right
theorem row10631_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10631 row10631_large_right :=
  Budget.RowUpperBound.branch row10631_large_right_left_covers row10631_large_right_right_covers

def row10631_large : List Nat := row10631_large_left ++ row10631_large_right
theorem row10631_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10631 row10631_large :=
  Budget.RowUpperBound.branch row10631_large_left_covers row10631_large_right_covers

def row10631 : List Nat := row10631_large
theorem row10631_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10631 row10631 := row10631_large_covers

def row10639_small : List Nat := []
theorem row10639_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10639 row10639_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10639_block_005 : List Nat := []
theorem row10639_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10639 row10639_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10639_block_006 : List Nat := [10891, 11491, 11503, 11587]
def plan10639_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 17), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .possible, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10639_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10639 row10639_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10639_block_006
  · decide
  · rfl

def row10639_block_007 : List Nat := [12433, 12517, 12637, 14431, 14461]
def plan10639_block_007 : List Budget.Choice := [(.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .second, (.firstFactor 29), (.secondFactor 29), .first, (.secondFactor 23), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 13), .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), .possible]
set_option maxRecDepth 2048 in
theorem row10639_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10639 row10639_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10639_block_007
  · decide
  · rfl

def row10639_block_008 : List Nat := [14827, 15271]
def plan10639_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .possible, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .second, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10639_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10639 row10639_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10639_block_008
  · decide
  · rfl

def row10639_block_009 : List Nat := []
theorem row10639_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10639 row10639_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10639_block_010 : List Nat := []
theorem row10639_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10639 row10639_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10639_block_011 : List Nat := []
theorem row10639_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10639 row10639_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10639_block_012 : List Nat := []
theorem row10639_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10639 row10639_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10639_block_013 : List Nat := []
theorem row10639_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10639 row10639_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10639_large_left_left : List Nat := row10639_small ++ row10639_block_005
theorem row10639_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10639 row10639_large_left_left :=
  Budget.RowUpperBound.branch row10639_small_covers row10639_block_005_covers

def row10639_large_left_right_right : List Nat := row10639_block_007 ++ row10639_block_008
theorem row10639_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10639 row10639_large_left_right_right :=
  Budget.RowUpperBound.branch row10639_block_007_covers row10639_block_008_covers

def row10639_large_left_right : List Nat := row10639_block_006 ++ row10639_large_left_right_right
theorem row10639_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10639 row10639_large_left_right :=
  Budget.RowUpperBound.branch row10639_block_006_covers row10639_large_left_right_right_covers

def row10639_large_left : List Nat := row10639_large_left_left ++ row10639_large_left_right
theorem row10639_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10639 row10639_large_left :=
  Budget.RowUpperBound.branch row10639_large_left_left_covers row10639_large_left_right_covers

def row10639_large_right_left : List Nat := row10639_block_009 ++ row10639_block_010
theorem row10639_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10639 row10639_large_right_left :=
  Budget.RowUpperBound.branch row10639_block_009_covers row10639_block_010_covers

def row10639_large_right_right_right : List Nat := row10639_block_012 ++ row10639_block_013
theorem row10639_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10639 row10639_large_right_right_right :=
  Budget.RowUpperBound.branch row10639_block_012_covers row10639_block_013_covers

def row10639_large_right_right : List Nat := row10639_block_011 ++ row10639_large_right_right_right
theorem row10639_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10639 row10639_large_right_right :=
  Budget.RowUpperBound.branch row10639_block_011_covers row10639_large_right_right_right_covers

def row10639_large_right : List Nat := row10639_large_right_left ++ row10639_large_right_right
theorem row10639_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10639 row10639_large_right :=
  Budget.RowUpperBound.branch row10639_large_right_left_covers row10639_large_right_right_covers

def row10639_large : List Nat := row10639_large_left ++ row10639_large_right
theorem row10639_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10639 row10639_large :=
  Budget.RowUpperBound.branch row10639_large_left_covers row10639_large_right_covers

def row10639 : List Nat := row10639_large
theorem row10639_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10639 row10639 := row10639_large_covers

end ProjectEulerSolutions.PrimePairLarge
