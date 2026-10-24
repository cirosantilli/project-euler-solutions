import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9551_small : List Nat := []
theorem row9551_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9551 row9551_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9551_block_005 : List Nat := []
def plan9551_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), .first, .first, .first, .second, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 37), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9551_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9551 row9551_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9551_block_005
  · decide
  · rfl

def row9551_block_006 : List Nat := [12401]
def plan9551_block_006 : List Budget.Choice := [.second, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .second, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .second, .second, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row9551_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9551 row9551_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9551_block_006
  · decide
  · rfl

def row9551_block_007 : List Nat := [12497, 13421]
def plan9551_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .second, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 19), .second, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .second, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9551_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9551 row9551_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9551_block_007
  · decide
  · rfl

def row9551_block_008 : List Nat := [15017, 15647]
def plan9551_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .possible, .first, .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 31), .second, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 13), .second, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9551_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9551 row9551_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9551_block_008
  · decide
  · rfl

def row9551_block_009 : List Nat := []
theorem row9551_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9551 row9551_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9551_block_010 : List Nat := []
theorem row9551_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9551 row9551_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9551_block_011 : List Nat := []
theorem row9551_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9551 row9551_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9551_block_012 : List Nat := []
theorem row9551_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9551 row9551_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9551_block_013 : List Nat := []
theorem row9551_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9551 row9551_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9551_large_left_left : List Nat := row9551_small ++ row9551_block_005
theorem row9551_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9551 row9551_large_left_left :=
  Budget.RowUpperBound.branch row9551_small_covers row9551_block_005_covers

def row9551_large_left_right_right : List Nat := row9551_block_007 ++ row9551_block_008
theorem row9551_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9551 row9551_large_left_right_right :=
  Budget.RowUpperBound.branch row9551_block_007_covers row9551_block_008_covers

def row9551_large_left_right : List Nat := row9551_block_006 ++ row9551_large_left_right_right
theorem row9551_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9551 row9551_large_left_right :=
  Budget.RowUpperBound.branch row9551_block_006_covers row9551_large_left_right_right_covers

def row9551_large_left : List Nat := row9551_large_left_left ++ row9551_large_left_right
theorem row9551_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9551 row9551_large_left :=
  Budget.RowUpperBound.branch row9551_large_left_left_covers row9551_large_left_right_covers

def row9551_large_right_left : List Nat := row9551_block_009 ++ row9551_block_010
theorem row9551_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9551 row9551_large_right_left :=
  Budget.RowUpperBound.branch row9551_block_009_covers row9551_block_010_covers

def row9551_large_right_right_right : List Nat := row9551_block_012 ++ row9551_block_013
theorem row9551_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9551 row9551_large_right_right_right :=
  Budget.RowUpperBound.branch row9551_block_012_covers row9551_block_013_covers

def row9551_large_right_right : List Nat := row9551_block_011 ++ row9551_large_right_right_right
theorem row9551_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9551 row9551_large_right_right :=
  Budget.RowUpperBound.branch row9551_block_011_covers row9551_large_right_right_right_covers

def row9551_large_right : List Nat := row9551_large_right_left ++ row9551_large_right_right
theorem row9551_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9551 row9551_large_right :=
  Budget.RowUpperBound.branch row9551_large_right_left_covers row9551_large_right_right_covers

def row9551_large : List Nat := row9551_large_left ++ row9551_large_right
theorem row9551_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9551 row9551_large :=
  Budget.RowUpperBound.branch row9551_large_left_covers row9551_large_right_covers

def row9551 : List Nat := row9551_large
theorem row9551_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9551 row9551 := row9551_large_covers

def row9587_small : List Nat := []
theorem row9587_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9587 row9587_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9587_block_005 : List Nat := []
def plan9587_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9587_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9587 row9587_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9587_block_005
  · decide
  · rfl

def row9587_block_006 : List Nat := [10847, 10979, 11483]
def plan9587_block_006 : List Budget.Choice := [(.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .possible, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .second, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9587_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9587 row9587_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9587_block_006
  · decide
  · rfl

def row9587_block_007 : List Nat := [13463, 14033]
def plan9587_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 29), (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 31), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9587_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9587 row9587_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9587_block_007
  · decide
  · rfl

def row9587_block_008 : List Nat := [15683, 16103, 16223]
def plan9587_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 29), (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .second, .first, .possible, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9587_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9587 row9587_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9587_block_008
  · decide
  · rfl

def row9587_block_009 : List Nat := []
theorem row9587_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9587 row9587_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9587_block_010 : List Nat := []
theorem row9587_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9587 row9587_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9587_block_011 : List Nat := []
theorem row9587_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9587 row9587_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9587_block_012 : List Nat := []
theorem row9587_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9587 row9587_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9587_block_013 : List Nat := []
theorem row9587_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9587 row9587_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9587_large_left_left : List Nat := row9587_small ++ row9587_block_005
theorem row9587_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9587 row9587_large_left_left :=
  Budget.RowUpperBound.branch row9587_small_covers row9587_block_005_covers

def row9587_large_left_right_right : List Nat := row9587_block_007 ++ row9587_block_008
theorem row9587_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9587 row9587_large_left_right_right :=
  Budget.RowUpperBound.branch row9587_block_007_covers row9587_block_008_covers

def row9587_large_left_right : List Nat := row9587_block_006 ++ row9587_large_left_right_right
theorem row9587_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9587 row9587_large_left_right :=
  Budget.RowUpperBound.branch row9587_block_006_covers row9587_large_left_right_right_covers

def row9587_large_left : List Nat := row9587_large_left_left ++ row9587_large_left_right
theorem row9587_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9587 row9587_large_left :=
  Budget.RowUpperBound.branch row9587_large_left_left_covers row9587_large_left_right_covers

def row9587_large_right_left : List Nat := row9587_block_009 ++ row9587_block_010
theorem row9587_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9587 row9587_large_right_left :=
  Budget.RowUpperBound.branch row9587_block_009_covers row9587_block_010_covers

def row9587_large_right_right_right : List Nat := row9587_block_012 ++ row9587_block_013
theorem row9587_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9587 row9587_large_right_right_right :=
  Budget.RowUpperBound.branch row9587_block_012_covers row9587_block_013_covers

def row9587_large_right_right : List Nat := row9587_block_011 ++ row9587_large_right_right_right
theorem row9587_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9587 row9587_large_right_right :=
  Budget.RowUpperBound.branch row9587_block_011_covers row9587_large_right_right_right_covers

def row9587_large_right : List Nat := row9587_large_right_left ++ row9587_large_right_right
theorem row9587_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9587 row9587_large_right :=
  Budget.RowUpperBound.branch row9587_large_right_left_covers row9587_large_right_right_covers

def row9587_large : List Nat := row9587_large_left ++ row9587_large_right
theorem row9587_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9587 row9587_large :=
  Budget.RowUpperBound.branch row9587_large_left_covers row9587_large_right_covers

def row9587 : List Nat := row9587_large
theorem row9587_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9587 row9587 := row9587_large_covers

def row9601_small : List Nat := []
theorem row9601_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9601 row9601_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9601_block_005 : List Nat := [10267]
def plan9601_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 19), .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9601_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9601 row9601_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9601_block_005
  · decide
  · rfl

def row9601_block_006 : List Nat := [10861, 11197, 11443, 11587]
def plan9601_block_006 : List Budget.Choice := [.first, .second, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 11), .second, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 19), .first, .first, (.firstFactor 37), (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9601_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9601 row9601_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9601_block_006
  · decide
  · rfl

def row9601_block_007 : List Nat := [14347]
def plan9601_block_007 : List Budget.Choice := [(.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 19), (.secondFactor 23), .second, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .second, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 31), .second, .first, (.firstFactor 7), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row9601_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9601 row9601_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9601_block_007
  · decide
  · rfl

def row9601_block_008 : List Nat := [14767, 14983, 15031, 15241, 15271, 15727, 15907]
def plan9601_block_008 : List Budget.Choice := [(.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .possible, (.firstFactor 11), .first, .possible, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, .possible, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .possible, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9601_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9601 row9601_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9601_block_008
  · decide
  · rfl

def row9601_block_009 : List Nat := []
theorem row9601_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9601 row9601_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9601_block_010 : List Nat := []
theorem row9601_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9601 row9601_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9601_block_011 : List Nat := []
theorem row9601_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9601 row9601_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9601_block_012 : List Nat := []
theorem row9601_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9601 row9601_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9601_block_013 : List Nat := []
theorem row9601_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9601 row9601_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9601_large_left_left : List Nat := row9601_small ++ row9601_block_005
theorem row9601_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9601 row9601_large_left_left :=
  Budget.RowUpperBound.branch row9601_small_covers row9601_block_005_covers

def row9601_large_left_right_right : List Nat := row9601_block_007 ++ row9601_block_008
theorem row9601_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9601 row9601_large_left_right_right :=
  Budget.RowUpperBound.branch row9601_block_007_covers row9601_block_008_covers

def row9601_large_left_right : List Nat := row9601_block_006 ++ row9601_large_left_right_right
theorem row9601_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9601 row9601_large_left_right :=
  Budget.RowUpperBound.branch row9601_block_006_covers row9601_large_left_right_right_covers

def row9601_large_left : List Nat := row9601_large_left_left ++ row9601_large_left_right
theorem row9601_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9601 row9601_large_left :=
  Budget.RowUpperBound.branch row9601_large_left_left_covers row9601_large_left_right_covers

def row9601_large_right_left : List Nat := row9601_block_009 ++ row9601_block_010
theorem row9601_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9601 row9601_large_right_left :=
  Budget.RowUpperBound.branch row9601_block_009_covers row9601_block_010_covers

def row9601_large_right_right_right : List Nat := row9601_block_012 ++ row9601_block_013
theorem row9601_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9601 row9601_large_right_right_right :=
  Budget.RowUpperBound.branch row9601_block_012_covers row9601_block_013_covers

def row9601_large_right_right : List Nat := row9601_block_011 ++ row9601_large_right_right_right
theorem row9601_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9601 row9601_large_right_right :=
  Budget.RowUpperBound.branch row9601_block_011_covers row9601_large_right_right_right_covers

def row9601_large_right : List Nat := row9601_large_right_left ++ row9601_large_right_right
theorem row9601_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9601 row9601_large_right :=
  Budget.RowUpperBound.branch row9601_large_right_left_covers row9601_large_right_right_covers

def row9601_large : List Nat := row9601_large_left ++ row9601_large_right
theorem row9601_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9601 row9601_large :=
  Budget.RowUpperBound.branch row9601_large_left_covers row9601_large_right_covers

def row9601 : List Nat := row9601_large
theorem row9601_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9601 row9601 := row9601_large_covers

def row9613_small : List Nat := []
theorem row9613_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9613 row9613_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9613_block_005 : List Nat := [9643, 9781, 9907, 10039]
def plan9613_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 31), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row9613_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9613 row9613_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9613_block_005
  · decide
  · rfl

def row9613_block_006 : List Nat := [10513, 10711, 10771, 11719, 12109]
def plan9613_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .second, .first, .first, .possible, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .possible, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9613_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9613 row9613_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9613_block_006
  · decide
  · rfl

def row9613_block_007 : List Nat := [12433, 12487, 12823, 12973, 13147, 13537, 14071, 14401, 14461]
def plan9613_block_007 : List Budget.Choice := [(.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .possible, (.secondFactor 13), (.secondFactor 7), (.firstFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .possible, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .possible, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .possible]
set_option maxRecDepth 2048 in
theorem row9613_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9613 row9613_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9613_block_007
  · decide
  · rfl

def row9613_block_008 : List Nat := [15601]
def plan9613_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .second, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .possible, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), (.firstFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9613_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9613 row9613_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9613_block_008
  · decide
  · rfl

def row9613_block_009 : List Nat := []
theorem row9613_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9613 row9613_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9613_block_010 : List Nat := []
theorem row9613_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9613 row9613_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9613_block_011 : List Nat := []
theorem row9613_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9613 row9613_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9613_block_012 : List Nat := []
theorem row9613_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9613 row9613_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9613_block_013 : List Nat := []
theorem row9613_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9613 row9613_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9613_large_left_left : List Nat := row9613_small ++ row9613_block_005
theorem row9613_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9613 row9613_large_left_left :=
  Budget.RowUpperBound.branch row9613_small_covers row9613_block_005_covers

def row9613_large_left_right_right : List Nat := row9613_block_007 ++ row9613_block_008
theorem row9613_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9613 row9613_large_left_right_right :=
  Budget.RowUpperBound.branch row9613_block_007_covers row9613_block_008_covers

def row9613_large_left_right : List Nat := row9613_block_006 ++ row9613_large_left_right_right
theorem row9613_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9613 row9613_large_left_right :=
  Budget.RowUpperBound.branch row9613_block_006_covers row9613_large_left_right_right_covers

def row9613_large_left : List Nat := row9613_large_left_left ++ row9613_large_left_right
theorem row9613_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9613 row9613_large_left :=
  Budget.RowUpperBound.branch row9613_large_left_left_covers row9613_large_left_right_covers

def row9613_large_right_left : List Nat := row9613_block_009 ++ row9613_block_010
theorem row9613_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9613 row9613_large_right_left :=
  Budget.RowUpperBound.branch row9613_block_009_covers row9613_block_010_covers

def row9613_large_right_right_right : List Nat := row9613_block_012 ++ row9613_block_013
theorem row9613_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9613 row9613_large_right_right_right :=
  Budget.RowUpperBound.branch row9613_block_012_covers row9613_block_013_covers

def row9613_large_right_right : List Nat := row9613_block_011 ++ row9613_large_right_right_right
theorem row9613_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9613 row9613_large_right_right :=
  Budget.RowUpperBound.branch row9613_block_011_covers row9613_large_right_right_right_covers

def row9613_large_right : List Nat := row9613_large_right_left ++ row9613_large_right_right
theorem row9613_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9613 row9613_large_right :=
  Budget.RowUpperBound.branch row9613_large_right_left_covers row9613_large_right_right_covers

def row9613_large : List Nat := row9613_large_left ++ row9613_large_right
theorem row9613_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9613 row9613_large :=
  Budget.RowUpperBound.branch row9613_large_left_covers row9613_large_right_covers

def row9613 : List Nat := row9613_large
theorem row9613_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9613 row9613 := row9613_large_covers

end ProjectEulerSolutions.PrimePairLarge
