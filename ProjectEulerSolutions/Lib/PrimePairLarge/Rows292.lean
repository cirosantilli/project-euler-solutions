import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9461_small : List Nat := []
theorem row9461_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9461 row9461_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9461_block_005 : List Nat := [9851, 10211]
def plan9461_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .possible, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), .possible, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9461_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9461 row9461_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9461_block_005
  · decide
  · rfl

def row9461_block_006 : List Nat := [10433, 10631, 11447, 11657]
def plan9461_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 37), .first, .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .second, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second]
set_option maxRecDepth 2048 in
theorem row9461_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9461 row9461_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9461_block_006
  · decide
  · rfl

def row9461_block_007 : List Nat := [12569, 13577]
def plan9461_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, (.secondFactor 11), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 31), (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9461_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9461 row9461_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9461_block_007
  · decide
  · rfl

def row9461_block_008 : List Nat := [14723, 15473, 16007, 16361]
def plan9461_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 7), .second, .first, (.firstFactor 31), .first, (.secondFactor 11), .possible, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 17), (.secondFactor 19), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, .second, .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9461_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9461 row9461_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9461_block_008
  · decide
  · rfl

def row9461_block_009 : List Nat := []
def plan9461_block_009 : List Budget.Choice := [.first, (.firstFactor 37), (.secondFactor 7), .second, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9461_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9461 row9461_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9461_block_009
  · decide
  · rfl

def row9461_block_010 : List Nat := []
theorem row9461_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9461 row9461_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9461_block_011 : List Nat := []
theorem row9461_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9461 row9461_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9461_block_012 : List Nat := []
theorem row9461_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9461 row9461_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9461_block_013 : List Nat := []
theorem row9461_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9461 row9461_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9461_large_left_left : List Nat := row9461_small ++ row9461_block_005
theorem row9461_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9461 row9461_large_left_left :=
  Budget.RowUpperBound.branch row9461_small_covers row9461_block_005_covers

def row9461_large_left_right_right : List Nat := row9461_block_007 ++ row9461_block_008
theorem row9461_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9461 row9461_large_left_right_right :=
  Budget.RowUpperBound.branch row9461_block_007_covers row9461_block_008_covers

def row9461_large_left_right : List Nat := row9461_block_006 ++ row9461_large_left_right_right
theorem row9461_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9461 row9461_large_left_right :=
  Budget.RowUpperBound.branch row9461_block_006_covers row9461_large_left_right_right_covers

def row9461_large_left : List Nat := row9461_large_left_left ++ row9461_large_left_right
theorem row9461_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9461 row9461_large_left :=
  Budget.RowUpperBound.branch row9461_large_left_left_covers row9461_large_left_right_covers

def row9461_large_right_left : List Nat := row9461_block_009 ++ row9461_block_010
theorem row9461_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9461 row9461_large_right_left :=
  Budget.RowUpperBound.branch row9461_block_009_covers row9461_block_010_covers

def row9461_large_right_right_right : List Nat := row9461_block_012 ++ row9461_block_013
theorem row9461_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9461 row9461_large_right_right_right :=
  Budget.RowUpperBound.branch row9461_block_012_covers row9461_block_013_covers

def row9461_large_right_right : List Nat := row9461_block_011 ++ row9461_large_right_right_right
theorem row9461_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9461 row9461_large_right_right :=
  Budget.RowUpperBound.branch row9461_block_011_covers row9461_large_right_right_right_covers

def row9461_large_right : List Nat := row9461_large_right_left ++ row9461_large_right_right
theorem row9461_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9461 row9461_large_right :=
  Budget.RowUpperBound.branch row9461_large_right_left_covers row9461_large_right_right_covers

def row9461_large : List Nat := row9461_large_left ++ row9461_large_right
theorem row9461_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9461 row9461_large :=
  Budget.RowUpperBound.branch row9461_large_left_covers row9461_large_right_covers

def row9461 : List Nat := row9461_large
theorem row9461_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9461 row9461 := row9461_large_covers

def row9463_small : List Nat := []
theorem row9463_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9463 row9463_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9463_block_005 : List Nat := [9661, 10093]
def plan9463_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9463_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9463 row9463_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9463_block_005
  · decide
  · rfl

def row9463_block_006 : List Nat := []
def plan9463_block_006 : List Budget.Choice := [.first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 23), .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 29), (.firstFactor 23), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9463_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9463 row9463_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9463_block_006
  · decide
  · rfl

def row9463_block_007 : List Nat := [12487, 12547, 12907, 12919, 13003, 13033, 13441, 14419]
def plan9463_block_007 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .possible, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), (.secondFactor 7), .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .second, (.firstFactor 17), .first, .first, .first, .possible, .first, (.secondFactor 13), .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, (.firstFactor 17), (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row9463_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9463 row9463_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9463_block_007
  · decide
  · rfl

def row9463_block_008 : List Nat := [14731, 15823, 16183]
def plan9463_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .second, .first, .first, .possible, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .possible, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 11), (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9463_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9463 row9463_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9463_block_008
  · decide
  · rfl

def row9463_block_009 : List Nat := [16561]
def plan9463_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9463_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9463 row9463_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9463_block_009
  · decide
  · rfl

def row9463_block_010 : List Nat := []
theorem row9463_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9463 row9463_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9463_block_011 : List Nat := []
theorem row9463_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9463 row9463_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9463_block_012 : List Nat := []
theorem row9463_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9463 row9463_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9463_block_013 : List Nat := []
theorem row9463_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9463 row9463_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9463_large_left_left : List Nat := row9463_small ++ row9463_block_005
theorem row9463_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9463 row9463_large_left_left :=
  Budget.RowUpperBound.branch row9463_small_covers row9463_block_005_covers

def row9463_large_left_right_right : List Nat := row9463_block_007 ++ row9463_block_008
theorem row9463_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9463 row9463_large_left_right_right :=
  Budget.RowUpperBound.branch row9463_block_007_covers row9463_block_008_covers

def row9463_large_left_right : List Nat := row9463_block_006 ++ row9463_large_left_right_right
theorem row9463_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9463 row9463_large_left_right :=
  Budget.RowUpperBound.branch row9463_block_006_covers row9463_large_left_right_right_covers

def row9463_large_left : List Nat := row9463_large_left_left ++ row9463_large_left_right
theorem row9463_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9463 row9463_large_left :=
  Budget.RowUpperBound.branch row9463_large_left_left_covers row9463_large_left_right_covers

def row9463_large_right_left : List Nat := row9463_block_009 ++ row9463_block_010
theorem row9463_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9463 row9463_large_right_left :=
  Budget.RowUpperBound.branch row9463_block_009_covers row9463_block_010_covers

def row9463_large_right_right_right : List Nat := row9463_block_012 ++ row9463_block_013
theorem row9463_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9463 row9463_large_right_right_right :=
  Budget.RowUpperBound.branch row9463_block_012_covers row9463_block_013_covers

def row9463_large_right_right : List Nat := row9463_block_011 ++ row9463_large_right_right_right
theorem row9463_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9463 row9463_large_right_right :=
  Budget.RowUpperBound.branch row9463_block_011_covers row9463_large_right_right_right_covers

def row9463_large_right : List Nat := row9463_large_right_left ++ row9463_large_right_right
theorem row9463_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9463 row9463_large_right :=
  Budget.RowUpperBound.branch row9463_large_right_left_covers row9463_large_right_right_covers

def row9463_large : List Nat := row9463_large_left ++ row9463_large_right
theorem row9463_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9463 row9463_large :=
  Budget.RowUpperBound.branch row9463_large_left_covers row9463_large_right_covers

def row9463 : List Nat := row9463_large
theorem row9463_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9463 row9463 := row9463_large_covers

def row9467_small : List Nat := []
theorem row9467_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9467 row9467_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9467_block_005 : List Nat := [9539, 9587, 10211, 10253]
def plan9467_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .possible, .first, .second, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .possible, (.secondFactor 11), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row9467_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9467 row9467_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9467_block_005
  · decide
  · rfl

def row9467_block_006 : List Nat := [11483, 11783, 11867, 12101, 12161, 12329]
def plan9467_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .second, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9467_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9467 row9467_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9467_block_006
  · decide
  · rfl

def row9467_block_007 : List Nat := [12539, 12653, 12953, 13451, 13751, 13841, 14081]
def plan9467_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .possible, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, .possible, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, .first, .second, (.secondFactor 11), (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9467_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9467 row9467_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9467_block_007
  · decide
  · rfl

def row9467_block_008 : List Nat := [14633]
def plan9467_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9467_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9467 row9467_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9467_block_008
  · decide
  · rfl

def row9467_block_009 : List Nat := []
def plan9467_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9467_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9467 row9467_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9467_block_009
  · decide
  · rfl

def row9467_block_010 : List Nat := []
theorem row9467_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9467 row9467_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9467_block_011 : List Nat := []
theorem row9467_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9467 row9467_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9467_block_012 : List Nat := []
theorem row9467_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9467 row9467_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9467_block_013 : List Nat := []
theorem row9467_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9467 row9467_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9467_large_left_left : List Nat := row9467_small ++ row9467_block_005
theorem row9467_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9467 row9467_large_left_left :=
  Budget.RowUpperBound.branch row9467_small_covers row9467_block_005_covers

def row9467_large_left_right_right : List Nat := row9467_block_007 ++ row9467_block_008
theorem row9467_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9467 row9467_large_left_right_right :=
  Budget.RowUpperBound.branch row9467_block_007_covers row9467_block_008_covers

def row9467_large_left_right : List Nat := row9467_block_006 ++ row9467_large_left_right_right
theorem row9467_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9467 row9467_large_left_right :=
  Budget.RowUpperBound.branch row9467_block_006_covers row9467_large_left_right_right_covers

def row9467_large_left : List Nat := row9467_large_left_left ++ row9467_large_left_right
theorem row9467_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9467 row9467_large_left :=
  Budget.RowUpperBound.branch row9467_large_left_left_covers row9467_large_left_right_covers

def row9467_large_right_left : List Nat := row9467_block_009 ++ row9467_block_010
theorem row9467_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9467 row9467_large_right_left :=
  Budget.RowUpperBound.branch row9467_block_009_covers row9467_block_010_covers

def row9467_large_right_right_right : List Nat := row9467_block_012 ++ row9467_block_013
theorem row9467_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9467 row9467_large_right_right_right :=
  Budget.RowUpperBound.branch row9467_block_012_covers row9467_block_013_covers

def row9467_large_right_right : List Nat := row9467_block_011 ++ row9467_large_right_right_right
theorem row9467_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9467 row9467_large_right_right :=
  Budget.RowUpperBound.branch row9467_block_011_covers row9467_large_right_right_right_covers

def row9467_large_right : List Nat := row9467_large_right_left ++ row9467_large_right_right
theorem row9467_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9467 row9467_large_right :=
  Budget.RowUpperBound.branch row9467_large_right_left_covers row9467_large_right_right_covers

def row9467_large : List Nat := row9467_large_left ++ row9467_large_right
theorem row9467_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9467 row9467_large :=
  Budget.RowUpperBound.branch row9467_large_left_covers row9467_large_right_covers

def row9467 : List Nat := row9467_large
theorem row9467_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9467 row9467 := row9467_large_covers

def row9473_small : List Nat := []
theorem row9473_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9473 row9473_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9473_block_005 : List Nat := []
def plan9473_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), (.secondFactor 13), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9473_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9473 row9473_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9473_block_005
  · decide
  · rfl

def row9473_block_006 : List Nat := [10799, 11597, 11789, 11969]
def plan9473_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 23), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .possible, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9473_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9473 row9473_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9473_block_006
  · decide
  · rfl

def row9473_block_007 : List Nat := [13421]
def plan9473_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9473_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9473 row9473_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9473_block_007
  · decide
  · rfl

def row9473_block_008 : List Nat := [15527, 15581, 15923]
def plan9473_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9473_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9473 row9473_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9473_block_008
  · decide
  · rfl

def row9473_block_009 : List Nat := []
def plan9473_block_009 : List Budget.Choice := [.first, .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9473_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9473 row9473_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9473_block_009
  · decide
  · rfl

def row9473_block_010 : List Nat := []
theorem row9473_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9473 row9473_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9473_block_011 : List Nat := []
theorem row9473_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9473 row9473_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9473_block_012 : List Nat := []
theorem row9473_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9473 row9473_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9473_block_013 : List Nat := []
theorem row9473_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9473 row9473_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9473_large_left_left : List Nat := row9473_small ++ row9473_block_005
theorem row9473_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9473 row9473_large_left_left :=
  Budget.RowUpperBound.branch row9473_small_covers row9473_block_005_covers

def row9473_large_left_right_right : List Nat := row9473_block_007 ++ row9473_block_008
theorem row9473_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9473 row9473_large_left_right_right :=
  Budget.RowUpperBound.branch row9473_block_007_covers row9473_block_008_covers

def row9473_large_left_right : List Nat := row9473_block_006 ++ row9473_large_left_right_right
theorem row9473_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9473 row9473_large_left_right :=
  Budget.RowUpperBound.branch row9473_block_006_covers row9473_large_left_right_right_covers

def row9473_large_left : List Nat := row9473_large_left_left ++ row9473_large_left_right
theorem row9473_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9473 row9473_large_left :=
  Budget.RowUpperBound.branch row9473_large_left_left_covers row9473_large_left_right_covers

def row9473_large_right_left : List Nat := row9473_block_009 ++ row9473_block_010
theorem row9473_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9473 row9473_large_right_left :=
  Budget.RowUpperBound.branch row9473_block_009_covers row9473_block_010_covers

def row9473_large_right_right_right : List Nat := row9473_block_012 ++ row9473_block_013
theorem row9473_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9473 row9473_large_right_right_right :=
  Budget.RowUpperBound.branch row9473_block_012_covers row9473_block_013_covers

def row9473_large_right_right : List Nat := row9473_block_011 ++ row9473_large_right_right_right
theorem row9473_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9473 row9473_large_right_right :=
  Budget.RowUpperBound.branch row9473_block_011_covers row9473_large_right_right_right_covers

def row9473_large_right : List Nat := row9473_large_right_left ++ row9473_large_right_right
theorem row9473_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9473 row9473_large_right :=
  Budget.RowUpperBound.branch row9473_large_right_left_covers row9473_large_right_right_covers

def row9473_large : List Nat := row9473_large_left ++ row9473_large_right
theorem row9473_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9473 row9473_large :=
  Budget.RowUpperBound.branch row9473_large_left_covers row9473_large_right_covers

def row9473 : List Nat := row9473_large
theorem row9473_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9473 row9473 := row9473_large_covers

end ProjectEulerSolutions.PrimePairLarge
