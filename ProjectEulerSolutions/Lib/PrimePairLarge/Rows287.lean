import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9293_small : List Nat := []
theorem row9293_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9293 row9293_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9293_block_005 : List Nat := [9311, 10139]
def plan9293_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .possible, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 23), (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .second, .first, (.firstFactor 29), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9293_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9293 row9293_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9293_block_005
  · decide
  · rfl

def row9293_block_006 : List Nat := [10979]
def plan9293_block_006 : List Budget.Choice := [.second, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 23), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row9293_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9293 row9293_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9293_block_006
  · decide
  · rfl

def row9293_block_007 : List Nat := [12647, 12899, 13001, 13241, 13337, 13907]
def plan9293_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .possible, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9293_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9293 row9293_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9293_block_007
  · decide
  · rfl

def row9293_block_008 : List Nat := [14627, 15641, 15671]
def plan9293_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 31), .possible, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 13), .second, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9293_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9293 row9293_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9293_block_008
  · decide
  · rfl

def row9293_block_009 : List Nat := []
def plan9293_block_009 : List Budget.Choice := [.first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9293_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9293 row9293_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9293_block_009
  · decide
  · rfl

def row9293_block_010 : List Nat := []
theorem row9293_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9293 row9293_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9293_block_011 : List Nat := []
theorem row9293_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9293 row9293_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9293_block_012 : List Nat := []
theorem row9293_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9293 row9293_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9293_block_013 : List Nat := []
theorem row9293_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9293 row9293_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9293_large_left_left : List Nat := row9293_small ++ row9293_block_005
theorem row9293_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9293 row9293_large_left_left :=
  Budget.RowUpperBound.branch row9293_small_covers row9293_block_005_covers

def row9293_large_left_right_right : List Nat := row9293_block_007 ++ row9293_block_008
theorem row9293_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9293 row9293_large_left_right_right :=
  Budget.RowUpperBound.branch row9293_block_007_covers row9293_block_008_covers

def row9293_large_left_right : List Nat := row9293_block_006 ++ row9293_large_left_right_right
theorem row9293_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9293 row9293_large_left_right :=
  Budget.RowUpperBound.branch row9293_block_006_covers row9293_large_left_right_right_covers

def row9293_large_left : List Nat := row9293_large_left_left ++ row9293_large_left_right
theorem row9293_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9293 row9293_large_left :=
  Budget.RowUpperBound.branch row9293_large_left_left_covers row9293_large_left_right_covers

def row9293_large_right_left : List Nat := row9293_block_009 ++ row9293_block_010
theorem row9293_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9293 row9293_large_right_left :=
  Budget.RowUpperBound.branch row9293_block_009_covers row9293_block_010_covers

def row9293_large_right_right_right : List Nat := row9293_block_012 ++ row9293_block_013
theorem row9293_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9293 row9293_large_right_right_right :=
  Budget.RowUpperBound.branch row9293_block_012_covers row9293_block_013_covers

def row9293_large_right_right : List Nat := row9293_block_011 ++ row9293_large_right_right_right
theorem row9293_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9293 row9293_large_right_right :=
  Budget.RowUpperBound.branch row9293_block_011_covers row9293_large_right_right_right_covers

def row9293_large_right : List Nat := row9293_large_right_left ++ row9293_large_right_right
theorem row9293_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9293 row9293_large_right :=
  Budget.RowUpperBound.branch row9293_large_right_left_covers row9293_large_right_right_covers

def row9293_large : List Nat := row9293_large_left ++ row9293_large_right
theorem row9293_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9293 row9293_large :=
  Budget.RowUpperBound.branch row9293_large_left_covers row9293_large_right_covers

def row9293 : List Nat := row9293_large
theorem row9293_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9293 row9293 := row9293_large_covers

def row9311_small : List Nat := []
theorem row9311_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9311 row9311_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9311_block_005 : List Nat := []
def plan9311_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9311_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9311 row9311_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9311_block_005
  · decide
  · rfl

def row9311_block_006 : List Nat := [10391, 11969]
def plan9311_block_006 : List Budget.Choice := [.possible, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .second, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9311_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9311 row9311_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9311_block_006
  · decide
  · rfl

def row9311_block_007 : List Nat := [12641, 13331, 13499, 13709, 14081, 14087]
def plan9311_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 7), .second, (.firstFactor 37), .second, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .possible, .first, .second, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .possible, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row9311_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9311 row9311_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9311_block_007
  · decide
  · rfl

def row9311_block_008 : List Nat := [15149, 15173, 15401]
def plan9311_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 17), .first, .first, .second, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .possible, (.secondFactor 17), .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9311_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9311 row9311_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9311_block_008
  · decide
  · rfl

def row9311_block_009 : List Nat := []
def plan9311_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, .second, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9311_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9311 row9311_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9311_block_009
  · decide
  · rfl

def row9311_block_010 : List Nat := []
theorem row9311_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9311 row9311_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9311_block_011 : List Nat := []
theorem row9311_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9311 row9311_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9311_block_012 : List Nat := []
theorem row9311_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9311 row9311_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9311_block_013 : List Nat := []
theorem row9311_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9311 row9311_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9311_large_left_left : List Nat := row9311_small ++ row9311_block_005
theorem row9311_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9311 row9311_large_left_left :=
  Budget.RowUpperBound.branch row9311_small_covers row9311_block_005_covers

def row9311_large_left_right_right : List Nat := row9311_block_007 ++ row9311_block_008
theorem row9311_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9311 row9311_large_left_right_right :=
  Budget.RowUpperBound.branch row9311_block_007_covers row9311_block_008_covers

def row9311_large_left_right : List Nat := row9311_block_006 ++ row9311_large_left_right_right
theorem row9311_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9311 row9311_large_left_right :=
  Budget.RowUpperBound.branch row9311_block_006_covers row9311_large_left_right_right_covers

def row9311_large_left : List Nat := row9311_large_left_left ++ row9311_large_left_right
theorem row9311_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9311 row9311_large_left :=
  Budget.RowUpperBound.branch row9311_large_left_left_covers row9311_large_left_right_covers

def row9311_large_right_left : List Nat := row9311_block_009 ++ row9311_block_010
theorem row9311_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9311 row9311_large_right_left :=
  Budget.RowUpperBound.branch row9311_block_009_covers row9311_block_010_covers

def row9311_large_right_right_right : List Nat := row9311_block_012 ++ row9311_block_013
theorem row9311_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9311 row9311_large_right_right_right :=
  Budget.RowUpperBound.branch row9311_block_012_covers row9311_block_013_covers

def row9311_large_right_right : List Nat := row9311_block_011 ++ row9311_large_right_right_right
theorem row9311_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9311 row9311_large_right_right :=
  Budget.RowUpperBound.branch row9311_block_011_covers row9311_large_right_right_right_covers

def row9311_large_right : List Nat := row9311_large_right_left ++ row9311_large_right_right
theorem row9311_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9311 row9311_large_right :=
  Budget.RowUpperBound.branch row9311_large_right_left_covers row9311_large_right_right_covers

def row9311_large : List Nat := row9311_large_left ++ row9311_large_right
theorem row9311_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9311 row9311_large :=
  Budget.RowUpperBound.branch row9311_large_left_covers row9311_large_right_covers

def row9311 : List Nat := row9311_large
theorem row9311_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9311 row9311 := row9311_large_covers

def row9319_small : List Nat := []
theorem row9319_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9319 row9319_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9319_block_005 : List Nat := []
def plan9319_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .second, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9319_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9319 row9319_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9319_block_005
  · decide
  · rfl

def row9319_block_006 : List Nat := [10513, 10597, 10753, 11131, 11173, 11551, 12097]
def plan9319_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .possible, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 19), .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row9319_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9319 row9319_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9319_block_006
  · decide
  · rfl

def row9319_block_007 : List Nat := [12829, 13249, 13399, 14407]
def plan9319_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .second, .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 31), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9319_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9319 row9319_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9319_block_007
  · decide
  · rfl

def row9319_block_008 : List Nat := [14593, 15619, 15661, 16057]
def plan9319_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, (.firstFactor 31), (.secondFactor 17), (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9319_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9319 row9319_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9319_block_008
  · decide
  · rfl

def row9319_block_009 : List Nat := []
def plan9319_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9319_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9319 row9319_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9319_block_009
  · decide
  · rfl

def row9319_block_010 : List Nat := []
theorem row9319_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9319 row9319_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9319_block_011 : List Nat := []
theorem row9319_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9319 row9319_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9319_block_012 : List Nat := []
theorem row9319_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9319 row9319_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9319_block_013 : List Nat := []
theorem row9319_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9319 row9319_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9319_large_left_left : List Nat := row9319_small ++ row9319_block_005
theorem row9319_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9319 row9319_large_left_left :=
  Budget.RowUpperBound.branch row9319_small_covers row9319_block_005_covers

def row9319_large_left_right_right : List Nat := row9319_block_007 ++ row9319_block_008
theorem row9319_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9319 row9319_large_left_right_right :=
  Budget.RowUpperBound.branch row9319_block_007_covers row9319_block_008_covers

def row9319_large_left_right : List Nat := row9319_block_006 ++ row9319_large_left_right_right
theorem row9319_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9319 row9319_large_left_right :=
  Budget.RowUpperBound.branch row9319_block_006_covers row9319_large_left_right_right_covers

def row9319_large_left : List Nat := row9319_large_left_left ++ row9319_large_left_right
theorem row9319_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9319 row9319_large_left :=
  Budget.RowUpperBound.branch row9319_large_left_left_covers row9319_large_left_right_covers

def row9319_large_right_left : List Nat := row9319_block_009 ++ row9319_block_010
theorem row9319_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9319 row9319_large_right_left :=
  Budget.RowUpperBound.branch row9319_block_009_covers row9319_block_010_covers

def row9319_large_right_right_right : List Nat := row9319_block_012 ++ row9319_block_013
theorem row9319_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9319 row9319_large_right_right_right :=
  Budget.RowUpperBound.branch row9319_block_012_covers row9319_block_013_covers

def row9319_large_right_right : List Nat := row9319_block_011 ++ row9319_large_right_right_right
theorem row9319_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9319 row9319_large_right_right :=
  Budget.RowUpperBound.branch row9319_block_011_covers row9319_large_right_right_right_covers

def row9319_large_right : List Nat := row9319_large_right_left ++ row9319_large_right_right
theorem row9319_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9319 row9319_large_right :=
  Budget.RowUpperBound.branch row9319_large_right_left_covers row9319_large_right_right_covers

def row9319_large : List Nat := row9319_large_left ++ row9319_large_right
theorem row9319_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9319 row9319_large :=
  Budget.RowUpperBound.branch row9319_large_left_covers row9319_large_right_covers

def row9319 : List Nat := row9319_large
theorem row9319_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9319 row9319 := row9319_large_covers

def row9323_small : List Nat := []
theorem row9323_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9323 row9323_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9323_block_005 : List Nat := [9491, 9533, 9551, 9791, 10331]
def plan9323_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 19), .possible, (.firstFactor 13), .first, .second, .possible, (.firstFactor 23), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9323_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9323 row9323_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9323_block_005
  · decide
  · rfl

def row9323_block_006 : List Nat := [10781, 11519, 12239]
def plan9323_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .second, .first, .first, .second, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9323_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9323 row9323_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9323_block_006
  · decide
  · rfl

def row9323_block_007 : List Nat := [13397, 13649, 14081]
def plan9323_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), (.firstFactor 23), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 17), .first, .second, .first, .possible, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 31), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9323_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9323 row9323_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9323_block_007
  · decide
  · rfl

def row9323_block_008 : List Nat := [14549, 14843, 15299, 16421]
def plan9323_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .second, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .second, .first, (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row9323_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9323 row9323_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9323_block_008
  · decide
  · rfl

def row9323_block_009 : List Nat := [16547]
def plan9323_block_009 : List Budget.Choice := [.first, (.firstFactor 19), .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9323_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9323 row9323_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9323_block_009
  · decide
  · rfl

def row9323_block_010 : List Nat := []
theorem row9323_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9323 row9323_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9323_block_011 : List Nat := []
theorem row9323_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9323 row9323_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9323_block_012 : List Nat := []
theorem row9323_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9323 row9323_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9323_block_013 : List Nat := []
theorem row9323_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9323 row9323_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9323_large_left_left : List Nat := row9323_small ++ row9323_block_005
theorem row9323_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9323 row9323_large_left_left :=
  Budget.RowUpperBound.branch row9323_small_covers row9323_block_005_covers

def row9323_large_left_right_right : List Nat := row9323_block_007 ++ row9323_block_008
theorem row9323_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9323 row9323_large_left_right_right :=
  Budget.RowUpperBound.branch row9323_block_007_covers row9323_block_008_covers

def row9323_large_left_right : List Nat := row9323_block_006 ++ row9323_large_left_right_right
theorem row9323_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9323 row9323_large_left_right :=
  Budget.RowUpperBound.branch row9323_block_006_covers row9323_large_left_right_right_covers

def row9323_large_left : List Nat := row9323_large_left_left ++ row9323_large_left_right
theorem row9323_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9323 row9323_large_left :=
  Budget.RowUpperBound.branch row9323_large_left_left_covers row9323_large_left_right_covers

def row9323_large_right_left : List Nat := row9323_block_009 ++ row9323_block_010
theorem row9323_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9323 row9323_large_right_left :=
  Budget.RowUpperBound.branch row9323_block_009_covers row9323_block_010_covers

def row9323_large_right_right_right : List Nat := row9323_block_012 ++ row9323_block_013
theorem row9323_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9323 row9323_large_right_right_right :=
  Budget.RowUpperBound.branch row9323_block_012_covers row9323_block_013_covers

def row9323_large_right_right : List Nat := row9323_block_011 ++ row9323_large_right_right_right
theorem row9323_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9323 row9323_large_right_right :=
  Budget.RowUpperBound.branch row9323_block_011_covers row9323_large_right_right_right_covers

def row9323_large_right : List Nat := row9323_large_right_left ++ row9323_large_right_right
theorem row9323_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9323 row9323_large_right :=
  Budget.RowUpperBound.branch row9323_large_right_left_covers row9323_large_right_right_covers

def row9323_large : List Nat := row9323_large_left ++ row9323_large_right
theorem row9323_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9323 row9323_large :=
  Budget.RowUpperBound.branch row9323_large_left_covers row9323_large_right_covers

def row9323 : List Nat := row9323_large
theorem row9323_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9323 row9323 := row9323_large_covers

end ProjectEulerSolutions.PrimePairLarge
