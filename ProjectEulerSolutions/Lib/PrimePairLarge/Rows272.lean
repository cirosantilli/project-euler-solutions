import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8747_small : List Nat := []
theorem row8747_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8747 row8747_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8747_block_005 : List Nat := [9377, 9479, 9851, 10139]
def plan8747_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .second, .possible, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8747_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8747 row8747_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8747_block_005
  · decide
  · rfl

def row8747_block_006 : List Nat := [10589, 10853, 11351, 12401]
def plan8747_block_006 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), .possible, (.firstFactor 11), .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 11), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 31), (.secondFactor 19), (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8747_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8747 row8747_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8747_block_006
  · decide
  · rfl

def row8747_block_007 : List Nat := [14033]
def plan8747_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 29), (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .second, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8747_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8747 row8747_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8747_block_007
  · decide
  · rfl

def row8747_block_008 : List Nat := [15227]
def plan8747_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .second, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8747_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8747 row8747_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8747_block_008
  · decide
  · rfl

def row8747_block_009 : List Nat := [16529]
def plan8747_block_009 : List Budget.Choice := [.first, .possible, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .second, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8747_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8747 row8747_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8747_block_009
  · decide
  · rfl

def row8747_block_010 : List Nat := []
theorem row8747_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8747 row8747_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8747_block_011 : List Nat := []
theorem row8747_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8747 row8747_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8747_block_012 : List Nat := []
theorem row8747_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8747 row8747_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8747_block_013 : List Nat := []
theorem row8747_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8747 row8747_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8747_large_left_left : List Nat := row8747_small ++ row8747_block_005
theorem row8747_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8747 row8747_large_left_left :=
  Budget.RowUpperBound.branch row8747_small_covers row8747_block_005_covers

def row8747_large_left_right_right : List Nat := row8747_block_007 ++ row8747_block_008
theorem row8747_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8747 row8747_large_left_right_right :=
  Budget.RowUpperBound.branch row8747_block_007_covers row8747_block_008_covers

def row8747_large_left_right : List Nat := row8747_block_006 ++ row8747_large_left_right_right
theorem row8747_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8747 row8747_large_left_right :=
  Budget.RowUpperBound.branch row8747_block_006_covers row8747_large_left_right_right_covers

def row8747_large_left : List Nat := row8747_large_left_left ++ row8747_large_left_right
theorem row8747_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8747 row8747_large_left :=
  Budget.RowUpperBound.branch row8747_large_left_left_covers row8747_large_left_right_covers

def row8747_large_right_left : List Nat := row8747_block_009 ++ row8747_block_010
theorem row8747_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8747 row8747_large_right_left :=
  Budget.RowUpperBound.branch row8747_block_009_covers row8747_block_010_covers

def row8747_large_right_right_right : List Nat := row8747_block_012 ++ row8747_block_013
theorem row8747_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8747 row8747_large_right_right_right :=
  Budget.RowUpperBound.branch row8747_block_012_covers row8747_block_013_covers

def row8747_large_right_right : List Nat := row8747_block_011 ++ row8747_large_right_right_right
theorem row8747_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8747 row8747_large_right_right :=
  Budget.RowUpperBound.branch row8747_block_011_covers row8747_large_right_right_right_covers

def row8747_large_right : List Nat := row8747_large_right_left ++ row8747_large_right_right
theorem row8747_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8747 row8747_large_right :=
  Budget.RowUpperBound.branch row8747_large_right_left_covers row8747_large_right_right_covers

def row8747_large : List Nat := row8747_large_left ++ row8747_large_right
theorem row8747_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8747 row8747_large :=
  Budget.RowUpperBound.branch row8747_large_left_covers row8747_large_right_covers

def row8747 : List Nat := row8747_large
theorem row8747_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8747 row8747 := row8747_large_covers

def row8753_small : List Nat := []
theorem row8753_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8753 row8753_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8753_block_005 : List Nat := [8819, 9041, 9533]
def plan8753_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 23), .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 37), .possible, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8753_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8753 row8753_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8753_block_005
  · decide
  · rfl

def row8753_block_006 : List Nat := [11279, 11777, 11903, 12203]
def plan8753_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, .possible, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .possible, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8753_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8753 row8753_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8753_block_006
  · decide
  · rfl

def row8753_block_007 : List Nat := [14081]
def plan8753_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 19), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .second, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8753_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8753 row8753_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8753_block_007
  · decide
  · rfl

def row8753_block_008 : List Nat := [14639, 14723, 15803, 16223]
def plan8753_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .second, .possible, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 31), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 13), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8753_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8753 row8753_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8753_block_008
  · decide
  · rfl

def row8753_block_009 : List Nat := [17021]
def plan8753_block_009 : List Budget.Choice := [.first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8753_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8753 row8753_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8753_block_009
  · decide
  · rfl

def row8753_block_010 : List Nat := []
theorem row8753_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8753 row8753_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8753_block_011 : List Nat := []
theorem row8753_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8753 row8753_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8753_block_012 : List Nat := []
theorem row8753_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8753 row8753_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8753_block_013 : List Nat := []
theorem row8753_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8753 row8753_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8753_large_left_left : List Nat := row8753_small ++ row8753_block_005
theorem row8753_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8753 row8753_large_left_left :=
  Budget.RowUpperBound.branch row8753_small_covers row8753_block_005_covers

def row8753_large_left_right_right : List Nat := row8753_block_007 ++ row8753_block_008
theorem row8753_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8753 row8753_large_left_right_right :=
  Budget.RowUpperBound.branch row8753_block_007_covers row8753_block_008_covers

def row8753_large_left_right : List Nat := row8753_block_006 ++ row8753_large_left_right_right
theorem row8753_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8753 row8753_large_left_right :=
  Budget.RowUpperBound.branch row8753_block_006_covers row8753_large_left_right_right_covers

def row8753_large_left : List Nat := row8753_large_left_left ++ row8753_large_left_right
theorem row8753_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8753 row8753_large_left :=
  Budget.RowUpperBound.branch row8753_large_left_left_covers row8753_large_left_right_covers

def row8753_large_right_left : List Nat := row8753_block_009 ++ row8753_block_010
theorem row8753_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8753 row8753_large_right_left :=
  Budget.RowUpperBound.branch row8753_block_009_covers row8753_block_010_covers

def row8753_large_right_right_right : List Nat := row8753_block_012 ++ row8753_block_013
theorem row8753_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8753 row8753_large_right_right_right :=
  Budget.RowUpperBound.branch row8753_block_012_covers row8753_block_013_covers

def row8753_large_right_right : List Nat := row8753_block_011 ++ row8753_large_right_right_right
theorem row8753_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8753 row8753_large_right_right :=
  Budget.RowUpperBound.branch row8753_block_011_covers row8753_large_right_right_right_covers

def row8753_large_right : List Nat := row8753_large_right_left ++ row8753_large_right_right
theorem row8753_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8753 row8753_large_right :=
  Budget.RowUpperBound.branch row8753_large_right_left_covers row8753_large_right_right_covers

def row8753_large : List Nat := row8753_large_left ++ row8753_large_right
theorem row8753_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8753 row8753_large :=
  Budget.RowUpperBound.branch row8753_large_left_covers row8753_large_right_covers

def row8753 : List Nat := row8753_large
theorem row8753_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8753 row8753 := row8753_large_covers

def row8761_small : List Nat := []
theorem row8761_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8761 row8761_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8761_block_005 : List Nat := [9613, 9619, 9787, 10267, 10321]
def plan8761_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .possible, .possible, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), .second, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .second, .second, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .second]
set_option maxRecDepth 2048 in
theorem row8761_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8761 row8761_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8761_block_005
  · decide
  · rfl

def row8761_block_006 : List Nat := [10789, 10993, 12373]
def plan8761_block_006 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 23), .first, .possible, .first, .first, (.firstFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8761_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8761 row8761_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8761_block_006
  · decide
  · rfl

def row8761_block_007 : List Nat := [12583, 13669, 14143]
def plan8761_block_007 : List Budget.Choice := [(.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8761_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8761 row8761_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8761_block_007
  · decide
  · rfl

def row8761_block_008 : List Nat := [14887, 15619, 16453]
def plan8761_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 19), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 23), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, (.firstFactor 29), .possible, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .second, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8761_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8761 row8761_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8761_block_008
  · decide
  · rfl

def row8761_block_009 : List Nat := []
def plan8761_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8761_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8761 row8761_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8761_block_009
  · decide
  · rfl

def row8761_block_010 : List Nat := []
theorem row8761_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8761 row8761_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8761_block_011 : List Nat := []
theorem row8761_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8761 row8761_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8761_block_012 : List Nat := []
theorem row8761_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8761 row8761_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8761_block_013 : List Nat := []
theorem row8761_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8761 row8761_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8761_large_left_left : List Nat := row8761_small ++ row8761_block_005
theorem row8761_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8761 row8761_large_left_left :=
  Budget.RowUpperBound.branch row8761_small_covers row8761_block_005_covers

def row8761_large_left_right_right : List Nat := row8761_block_007 ++ row8761_block_008
theorem row8761_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8761 row8761_large_left_right_right :=
  Budget.RowUpperBound.branch row8761_block_007_covers row8761_block_008_covers

def row8761_large_left_right : List Nat := row8761_block_006 ++ row8761_large_left_right_right
theorem row8761_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8761 row8761_large_left_right :=
  Budget.RowUpperBound.branch row8761_block_006_covers row8761_large_left_right_right_covers

def row8761_large_left : List Nat := row8761_large_left_left ++ row8761_large_left_right
theorem row8761_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8761 row8761_large_left :=
  Budget.RowUpperBound.branch row8761_large_left_left_covers row8761_large_left_right_covers

def row8761_large_right_left : List Nat := row8761_block_009 ++ row8761_block_010
theorem row8761_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8761 row8761_large_right_left :=
  Budget.RowUpperBound.branch row8761_block_009_covers row8761_block_010_covers

def row8761_large_right_right_right : List Nat := row8761_block_012 ++ row8761_block_013
theorem row8761_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8761 row8761_large_right_right_right :=
  Budget.RowUpperBound.branch row8761_block_012_covers row8761_block_013_covers

def row8761_large_right_right : List Nat := row8761_block_011 ++ row8761_large_right_right_right
theorem row8761_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8761 row8761_large_right_right :=
  Budget.RowUpperBound.branch row8761_block_011_covers row8761_large_right_right_right_covers

def row8761_large_right : List Nat := row8761_large_right_left ++ row8761_large_right_right
theorem row8761_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8761 row8761_large_right :=
  Budget.RowUpperBound.branch row8761_large_right_left_covers row8761_large_right_right_covers

def row8761_large : List Nat := row8761_large_left ++ row8761_large_right
theorem row8761_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8761 row8761_large :=
  Budget.RowUpperBound.branch row8761_large_left_covers row8761_large_right_covers

def row8761 : List Nat := row8761_large
theorem row8761_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8761 row8761 := row8761_large_covers

def row8779_small : List Nat := []
theorem row8779_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8779 row8779_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8779_block_005 : List Nat := [9697]
def plan8779_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .second, .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8779_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8779 row8779_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8779_block_005
  · decide
  · rfl

def row8779_block_006 : List Nat := [11071]
def plan8779_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .second, .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8779_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8779 row8779_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8779_block_006
  · decide
  · rfl

def row8779_block_007 : List Nat := [12823, 13063, 13567, 13693]
def plan8779_block_007 : List Budget.Choice := [(.secondFactor 17), .second, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .possible, (.firstFactor 23), (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 17), .first, .first, .first, .possible, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 37), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row8779_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8779 row8779_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8779_block_007
  · decide
  · rfl

def row8779_block_008 : List Nat := [15823, 16339]
def plan8779_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .possible, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 31), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 19), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8779_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8779 row8779_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8779_block_008
  · decide
  · rfl

def row8779_block_009 : List Nat := [16759, 17041]
def plan8779_block_009 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .second, (.firstFactor 13), (.secondFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 31), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8779_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8779 row8779_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8779_block_009
  · decide
  · rfl

def row8779_block_010 : List Nat := []
theorem row8779_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8779 row8779_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8779_block_011 : List Nat := []
theorem row8779_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8779 row8779_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8779_block_012 : List Nat := []
theorem row8779_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8779 row8779_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8779_block_013 : List Nat := []
theorem row8779_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8779 row8779_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8779_large_left_left : List Nat := row8779_small ++ row8779_block_005
theorem row8779_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8779 row8779_large_left_left :=
  Budget.RowUpperBound.branch row8779_small_covers row8779_block_005_covers

def row8779_large_left_right_right : List Nat := row8779_block_007 ++ row8779_block_008
theorem row8779_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8779 row8779_large_left_right_right :=
  Budget.RowUpperBound.branch row8779_block_007_covers row8779_block_008_covers

def row8779_large_left_right : List Nat := row8779_block_006 ++ row8779_large_left_right_right
theorem row8779_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8779 row8779_large_left_right :=
  Budget.RowUpperBound.branch row8779_block_006_covers row8779_large_left_right_right_covers

def row8779_large_left : List Nat := row8779_large_left_left ++ row8779_large_left_right
theorem row8779_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8779 row8779_large_left :=
  Budget.RowUpperBound.branch row8779_large_left_left_covers row8779_large_left_right_covers

def row8779_large_right_left : List Nat := row8779_block_009 ++ row8779_block_010
theorem row8779_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8779 row8779_large_right_left :=
  Budget.RowUpperBound.branch row8779_block_009_covers row8779_block_010_covers

def row8779_large_right_right_right : List Nat := row8779_block_012 ++ row8779_block_013
theorem row8779_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8779 row8779_large_right_right_right :=
  Budget.RowUpperBound.branch row8779_block_012_covers row8779_block_013_covers

def row8779_large_right_right : List Nat := row8779_block_011 ++ row8779_large_right_right_right
theorem row8779_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8779 row8779_large_right_right :=
  Budget.RowUpperBound.branch row8779_block_011_covers row8779_large_right_right_right_covers

def row8779_large_right : List Nat := row8779_large_right_left ++ row8779_large_right_right
theorem row8779_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8779 row8779_large_right :=
  Budget.RowUpperBound.branch row8779_large_right_left_covers row8779_large_right_right_covers

def row8779_large : List Nat := row8779_large_left ++ row8779_large_right
theorem row8779_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8779 row8779_large :=
  Budget.RowUpperBound.branch row8779_large_left_covers row8779_large_right_covers

def row8779 : List Nat := row8779_large
theorem row8779_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8779 row8779 := row8779_large_covers

end ProjectEulerSolutions.PrimePairLarge
