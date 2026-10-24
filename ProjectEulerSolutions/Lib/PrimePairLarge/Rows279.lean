import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9007_small : List Nat := []
theorem row9007_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9007 row9007_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9007_block_005 : List Nat := [9631, 10009]
def plan9007_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 37), (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .second, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9007_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9007 row9007_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9007_block_005
  · decide
  · rfl

def row9007_block_006 : List Nat := [11689]
def plan9007_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.secondFactor 11), .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row9007_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9007 row9007_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9007_block_006
  · decide
  · rfl

def row9007_block_007 : List Nat := [13399]
def plan9007_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 13), .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .second, (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .possible, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .second, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9007_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9007 row9007_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9007_block_007
  · decide
  · rfl

def row9007_block_008 : List Nat := [14923, 15073, 15427, 15493]
def plan9007_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 7), .possible, .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .second, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .second, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9007_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9007 row9007_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9007_block_008
  · decide
  · rfl

def row9007_block_009 : List Nat := [16759]
def plan9007_block_009 : List Budget.Choice := [.second, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9007_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9007 row9007_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9007_block_009
  · decide
  · rfl

def row9007_block_010 : List Nat := []
theorem row9007_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9007 row9007_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9007_block_011 : List Nat := []
theorem row9007_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9007 row9007_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9007_block_012 : List Nat := []
theorem row9007_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9007 row9007_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9007_block_013 : List Nat := []
theorem row9007_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9007 row9007_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9007_large_left_left : List Nat := row9007_small ++ row9007_block_005
theorem row9007_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9007 row9007_large_left_left :=
  Budget.RowUpperBound.branch row9007_small_covers row9007_block_005_covers

def row9007_large_left_right_right : List Nat := row9007_block_007 ++ row9007_block_008
theorem row9007_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9007 row9007_large_left_right_right :=
  Budget.RowUpperBound.branch row9007_block_007_covers row9007_block_008_covers

def row9007_large_left_right : List Nat := row9007_block_006 ++ row9007_large_left_right_right
theorem row9007_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9007 row9007_large_left_right :=
  Budget.RowUpperBound.branch row9007_block_006_covers row9007_large_left_right_right_covers

def row9007_large_left : List Nat := row9007_large_left_left ++ row9007_large_left_right
theorem row9007_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9007 row9007_large_left :=
  Budget.RowUpperBound.branch row9007_large_left_left_covers row9007_large_left_right_covers

def row9007_large_right_left : List Nat := row9007_block_009 ++ row9007_block_010
theorem row9007_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9007 row9007_large_right_left :=
  Budget.RowUpperBound.branch row9007_block_009_covers row9007_block_010_covers

def row9007_large_right_right_right : List Nat := row9007_block_012 ++ row9007_block_013
theorem row9007_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9007 row9007_large_right_right_right :=
  Budget.RowUpperBound.branch row9007_block_012_covers row9007_block_013_covers

def row9007_large_right_right : List Nat := row9007_block_011 ++ row9007_large_right_right_right
theorem row9007_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9007 row9007_large_right_right :=
  Budget.RowUpperBound.branch row9007_block_011_covers row9007_large_right_right_right_covers

def row9007_large_right : List Nat := row9007_large_right_left ++ row9007_large_right_right
theorem row9007_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9007 row9007_large_right :=
  Budget.RowUpperBound.branch row9007_large_right_left_covers row9007_large_right_right_covers

def row9007_large : List Nat := row9007_large_left ++ row9007_large_right
theorem row9007_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9007 row9007_large :=
  Budget.RowUpperBound.branch row9007_large_left_covers row9007_large_right_covers

def row9007 : List Nat := row9007_large
theorem row9007_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9007 row9007 := row9007_large_covers

def row9011_small : List Nat := []
theorem row9011_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9011 row9011_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9011_block_005 : List Nat := [9209, 9311, 9749]
def plan9011_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .possible, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 17), .second, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 11), .first, .second, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9011_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9011 row9011_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9011_block_005
  · decide
  · rfl

def row9011_block_006 : List Nat := [10589, 11087, 11657]
def plan9011_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .possible, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .second, .possible, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9011_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9011 row9011_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9011_block_006
  · decide
  · rfl

def row9011_block_007 : List Nat := [12503]
def plan9011_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 31), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 29), .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row9011_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9011 row9011_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9011_block_007
  · decide
  · rfl

def row9011_block_008 : List Nat := [14561, 15359, 15377]
def plan9011_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .second, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .possible, .first, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .second]
set_option maxRecDepth 2048 in
theorem row9011_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9011 row9011_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9011_block_008
  · decide
  · rfl

def row9011_block_009 : List Nat := [16673, 16829]
def plan9011_block_009 : List Budget.Choice := [.first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9011_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9011 row9011_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9011_block_009
  · decide
  · rfl

def row9011_block_010 : List Nat := []
theorem row9011_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9011 row9011_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9011_block_011 : List Nat := []
theorem row9011_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9011 row9011_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9011_block_012 : List Nat := []
theorem row9011_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9011 row9011_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9011_block_013 : List Nat := []
theorem row9011_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9011 row9011_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9011_large_left_left : List Nat := row9011_small ++ row9011_block_005
theorem row9011_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9011 row9011_large_left_left :=
  Budget.RowUpperBound.branch row9011_small_covers row9011_block_005_covers

def row9011_large_left_right_right : List Nat := row9011_block_007 ++ row9011_block_008
theorem row9011_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9011 row9011_large_left_right_right :=
  Budget.RowUpperBound.branch row9011_block_007_covers row9011_block_008_covers

def row9011_large_left_right : List Nat := row9011_block_006 ++ row9011_large_left_right_right
theorem row9011_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9011 row9011_large_left_right :=
  Budget.RowUpperBound.branch row9011_block_006_covers row9011_large_left_right_right_covers

def row9011_large_left : List Nat := row9011_large_left_left ++ row9011_large_left_right
theorem row9011_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9011 row9011_large_left :=
  Budget.RowUpperBound.branch row9011_large_left_left_covers row9011_large_left_right_covers

def row9011_large_right_left : List Nat := row9011_block_009 ++ row9011_block_010
theorem row9011_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9011 row9011_large_right_left :=
  Budget.RowUpperBound.branch row9011_block_009_covers row9011_block_010_covers

def row9011_large_right_right_right : List Nat := row9011_block_012 ++ row9011_block_013
theorem row9011_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9011 row9011_large_right_right_right :=
  Budget.RowUpperBound.branch row9011_block_012_covers row9011_block_013_covers

def row9011_large_right_right : List Nat := row9011_block_011 ++ row9011_large_right_right_right
theorem row9011_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9011 row9011_large_right_right :=
  Budget.RowUpperBound.branch row9011_block_011_covers row9011_large_right_right_right_covers

def row9011_large_right : List Nat := row9011_large_right_left ++ row9011_large_right_right
theorem row9011_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9011 row9011_large_right :=
  Budget.RowUpperBound.branch row9011_large_right_left_covers row9011_large_right_right_covers

def row9011_large : List Nat := row9011_large_left ++ row9011_large_right
theorem row9011_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9011 row9011_large :=
  Budget.RowUpperBound.branch row9011_large_left_covers row9011_large_right_covers

def row9011 : List Nat := row9011_large
theorem row9011_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9011 row9011 := row9011_large_covers

def row9013_small : List Nat := []
theorem row9013_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9013 row9013_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9013_block_005 : List Nat := [9199, 9241, 9349]
def plan9013_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9013_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9013 row9013_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9013_block_005
  · decide
  · rfl

def row9013_block_006 : List Nat := []
def plan9013_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9013_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9013 row9013_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9013_block_006
  · decide
  · rfl

def row9013_block_007 : List Nat := [13099, 13267]
def plan9013_block_007 : List Budget.Choice := [.second, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .second, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 23), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row9013_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9013 row9013_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9013_block_007
  · decide
  · rfl

def row9013_block_008 : List Nat := [15307, 15361, 16231]
def plan9013_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .second, .second, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9013_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9013 row9013_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9013_block_008
  · decide
  · rfl

def row9013_block_009 : List Nat := []
def plan9013_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9013_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9013 row9013_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9013_block_009
  · decide
  · rfl

def row9013_block_010 : List Nat := []
theorem row9013_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9013 row9013_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9013_block_011 : List Nat := []
theorem row9013_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9013 row9013_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9013_block_012 : List Nat := []
theorem row9013_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9013 row9013_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9013_block_013 : List Nat := []
theorem row9013_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9013 row9013_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9013_large_left_left : List Nat := row9013_small ++ row9013_block_005
theorem row9013_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9013 row9013_large_left_left :=
  Budget.RowUpperBound.branch row9013_small_covers row9013_block_005_covers

def row9013_large_left_right_right : List Nat := row9013_block_007 ++ row9013_block_008
theorem row9013_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9013 row9013_large_left_right_right :=
  Budget.RowUpperBound.branch row9013_block_007_covers row9013_block_008_covers

def row9013_large_left_right : List Nat := row9013_block_006 ++ row9013_large_left_right_right
theorem row9013_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9013 row9013_large_left_right :=
  Budget.RowUpperBound.branch row9013_block_006_covers row9013_large_left_right_right_covers

def row9013_large_left : List Nat := row9013_large_left_left ++ row9013_large_left_right
theorem row9013_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9013 row9013_large_left :=
  Budget.RowUpperBound.branch row9013_large_left_left_covers row9013_large_left_right_covers

def row9013_large_right_left : List Nat := row9013_block_009 ++ row9013_block_010
theorem row9013_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9013 row9013_large_right_left :=
  Budget.RowUpperBound.branch row9013_block_009_covers row9013_block_010_covers

def row9013_large_right_right_right : List Nat := row9013_block_012 ++ row9013_block_013
theorem row9013_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9013 row9013_large_right_right_right :=
  Budget.RowUpperBound.branch row9013_block_012_covers row9013_block_013_covers

def row9013_large_right_right : List Nat := row9013_block_011 ++ row9013_large_right_right_right
theorem row9013_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9013 row9013_large_right_right :=
  Budget.RowUpperBound.branch row9013_block_011_covers row9013_large_right_right_right_covers

def row9013_large_right : List Nat := row9013_large_right_left ++ row9013_large_right_right
theorem row9013_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9013 row9013_large_right :=
  Budget.RowUpperBound.branch row9013_large_right_left_covers row9013_large_right_right_covers

def row9013_large : List Nat := row9013_large_left ++ row9013_large_right
theorem row9013_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9013 row9013_large :=
  Budget.RowUpperBound.branch row9013_large_left_covers row9013_large_right_covers

def row9013 : List Nat := row9013_large
theorem row9013_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9013 row9013 := row9013_large_covers

def row9029_small : List Nat := []
theorem row9029_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9029 row9029_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9029_block_005 : List Nat := [9281, 9491, 9743]
def plan9029_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .possible, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, .possible, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 31), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9029_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9029 row9029_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9029_block_005
  · decide
  · rfl

def row9029_block_006 : List Nat := [10391, 10847, 11927, 12011]
def plan9029_block_006 : List Budget.Choice := [.possible, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 13), (.secondFactor 31), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9029_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9029 row9029_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9029_block_006
  · decide
  · rfl

def row9029_block_007 : List Nat := [14249]
def plan9029_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9029_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9029 row9029_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9029_block_007
  · decide
  · rfl

def row9029_block_008 : List Nat := [14717, 15467, 15581, 15737]
def plan9029_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 19), .second, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 23), .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 23), .first, .first, (.secondFactor 17), (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9029_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9029 row9029_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9029_block_008
  · decide
  · rfl

def row9029_block_009 : List Nat := [16691, 16829]
def plan9029_block_009 : List Budget.Choice := [.first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 23), .possible, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 19), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9029_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9029 row9029_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9029_block_009
  · decide
  · rfl

def row9029_block_010 : List Nat := []
theorem row9029_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9029 row9029_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9029_block_011 : List Nat := []
theorem row9029_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9029 row9029_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9029_block_012 : List Nat := []
theorem row9029_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9029 row9029_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9029_block_013 : List Nat := []
theorem row9029_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9029 row9029_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9029_large_left_left : List Nat := row9029_small ++ row9029_block_005
theorem row9029_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9029 row9029_large_left_left :=
  Budget.RowUpperBound.branch row9029_small_covers row9029_block_005_covers

def row9029_large_left_right_right : List Nat := row9029_block_007 ++ row9029_block_008
theorem row9029_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9029 row9029_large_left_right_right :=
  Budget.RowUpperBound.branch row9029_block_007_covers row9029_block_008_covers

def row9029_large_left_right : List Nat := row9029_block_006 ++ row9029_large_left_right_right
theorem row9029_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9029 row9029_large_left_right :=
  Budget.RowUpperBound.branch row9029_block_006_covers row9029_large_left_right_right_covers

def row9029_large_left : List Nat := row9029_large_left_left ++ row9029_large_left_right
theorem row9029_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9029 row9029_large_left :=
  Budget.RowUpperBound.branch row9029_large_left_left_covers row9029_large_left_right_covers

def row9029_large_right_left : List Nat := row9029_block_009 ++ row9029_block_010
theorem row9029_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9029 row9029_large_right_left :=
  Budget.RowUpperBound.branch row9029_block_009_covers row9029_block_010_covers

def row9029_large_right_right_right : List Nat := row9029_block_012 ++ row9029_block_013
theorem row9029_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9029 row9029_large_right_right_right :=
  Budget.RowUpperBound.branch row9029_block_012_covers row9029_block_013_covers

def row9029_large_right_right : List Nat := row9029_block_011 ++ row9029_large_right_right_right
theorem row9029_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9029 row9029_large_right_right :=
  Budget.RowUpperBound.branch row9029_block_011_covers row9029_large_right_right_right_covers

def row9029_large_right : List Nat := row9029_large_right_left ++ row9029_large_right_right
theorem row9029_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9029 row9029_large_right :=
  Budget.RowUpperBound.branch row9029_large_right_left_covers row9029_large_right_right_covers

def row9029_large : List Nat := row9029_large_left ++ row9029_large_right
theorem row9029_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9029 row9029_large :=
  Budget.RowUpperBound.branch row9029_large_left_covers row9029_large_right_covers

def row9029 : List Nat := row9029_large
theorem row9029_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9029 row9029 := row9029_large_covers

end ProjectEulerSolutions.PrimePairLarge
