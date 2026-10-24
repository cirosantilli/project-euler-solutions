import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9787_small : List Nat := []
theorem row9787_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9787 row9787_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9787_block_005 : List Nat := []
def plan9787_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 31), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9787_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9787 row9787_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9787_block_005
  · decide
  · rfl

def row9787_block_006 : List Nat := [10867, 12391]
def plan9787_block_006 : List Budget.Choice := [.first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 29), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .possible, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9787_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9787 row9787_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9787_block_006
  · decide
  · rfl

def row9787_block_007 : List Nat := [12547, 13003, 13033, 13147, 13831, 14029, 14449]
def plan9787_block_007 : List Budget.Choice := [.second, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .possible, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .possible, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, .possible, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9787_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9787 row9787_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9787_block_007
  · decide
  · rfl

def row9787_block_008 : List Nat := [16111]
def plan9787_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9787_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9787 row9787_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9787_block_008
  · decide
  · rfl

def row9787_block_009 : List Nat := []
theorem row9787_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9787 row9787_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9787_block_010 : List Nat := []
theorem row9787_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9787 row9787_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9787_block_011 : List Nat := []
theorem row9787_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9787 row9787_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9787_block_012 : List Nat := []
theorem row9787_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9787 row9787_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9787_block_013 : List Nat := []
theorem row9787_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9787 row9787_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9787_large_left_left : List Nat := row9787_small ++ row9787_block_005
theorem row9787_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9787 row9787_large_left_left :=
  Budget.RowUpperBound.branch row9787_small_covers row9787_block_005_covers

def row9787_large_left_right_right : List Nat := row9787_block_007 ++ row9787_block_008
theorem row9787_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9787 row9787_large_left_right_right :=
  Budget.RowUpperBound.branch row9787_block_007_covers row9787_block_008_covers

def row9787_large_left_right : List Nat := row9787_block_006 ++ row9787_large_left_right_right
theorem row9787_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9787 row9787_large_left_right :=
  Budget.RowUpperBound.branch row9787_block_006_covers row9787_large_left_right_right_covers

def row9787_large_left : List Nat := row9787_large_left_left ++ row9787_large_left_right
theorem row9787_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9787 row9787_large_left :=
  Budget.RowUpperBound.branch row9787_large_left_left_covers row9787_large_left_right_covers

def row9787_large_right_left : List Nat := row9787_block_009 ++ row9787_block_010
theorem row9787_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9787 row9787_large_right_left :=
  Budget.RowUpperBound.branch row9787_block_009_covers row9787_block_010_covers

def row9787_large_right_right_right : List Nat := row9787_block_012 ++ row9787_block_013
theorem row9787_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9787 row9787_large_right_right_right :=
  Budget.RowUpperBound.branch row9787_block_012_covers row9787_block_013_covers

def row9787_large_right_right : List Nat := row9787_block_011 ++ row9787_large_right_right_right
theorem row9787_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9787 row9787_large_right_right :=
  Budget.RowUpperBound.branch row9787_block_011_covers row9787_large_right_right_right_covers

def row9787_large_right : List Nat := row9787_large_right_left ++ row9787_large_right_right
theorem row9787_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9787 row9787_large_right :=
  Budget.RowUpperBound.branch row9787_large_right_left_covers row9787_large_right_right_covers

def row9787_large : List Nat := row9787_large_left ++ row9787_large_right
theorem row9787_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9787 row9787_large :=
  Budget.RowUpperBound.branch row9787_large_left_covers row9787_large_right_covers

def row9787 : List Nat := row9787_large
theorem row9787_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9787 row9787 := row9787_large_covers

def row9791_small : List Nat := []
theorem row9791_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9791 row9791_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9791_block_005 : List Nat := [10103]
def plan9791_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9791_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9791 row9791_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9791_block_005
  · decide
  · rfl

def row9791_block_006 : List Nat := [10739, 10949, 11261, 11321, 11699, 12041, 12269]
def plan9791_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 37), .first, (.secondFactor 31), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 23), (.firstFactor 31), .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .second, .first, .first, (.firstFactor 31), .first, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row9791_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9791 row9791_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9791_block_006
  · decide
  · rfl

def row9791_block_007 : List Nat := [12479, 13313, 13763, 13799, 14051, 14177]
def plan9791_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .possible, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .second, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 23), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row9791_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9791 row9791_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9791_block_007
  · decide
  · rfl

def row9791_block_008 : List Nat := [14489, 15149]
def plan9791_block_008 : List Budget.Choice := [.first, .possible, .first, (.secondFactor 11), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .second, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .second, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9791_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9791 row9791_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9791_block_008
  · decide
  · rfl

def row9791_block_009 : List Nat := []
theorem row9791_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9791 row9791_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9791_block_010 : List Nat := []
theorem row9791_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9791 row9791_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9791_block_011 : List Nat := []
theorem row9791_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9791 row9791_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9791_block_012 : List Nat := []
theorem row9791_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9791 row9791_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9791_block_013 : List Nat := []
theorem row9791_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9791 row9791_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9791_large_left_left : List Nat := row9791_small ++ row9791_block_005
theorem row9791_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9791 row9791_large_left_left :=
  Budget.RowUpperBound.branch row9791_small_covers row9791_block_005_covers

def row9791_large_left_right_right : List Nat := row9791_block_007 ++ row9791_block_008
theorem row9791_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9791 row9791_large_left_right_right :=
  Budget.RowUpperBound.branch row9791_block_007_covers row9791_block_008_covers

def row9791_large_left_right : List Nat := row9791_block_006 ++ row9791_large_left_right_right
theorem row9791_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9791 row9791_large_left_right :=
  Budget.RowUpperBound.branch row9791_block_006_covers row9791_large_left_right_right_covers

def row9791_large_left : List Nat := row9791_large_left_left ++ row9791_large_left_right
theorem row9791_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9791 row9791_large_left :=
  Budget.RowUpperBound.branch row9791_large_left_left_covers row9791_large_left_right_covers

def row9791_large_right_left : List Nat := row9791_block_009 ++ row9791_block_010
theorem row9791_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9791 row9791_large_right_left :=
  Budget.RowUpperBound.branch row9791_block_009_covers row9791_block_010_covers

def row9791_large_right_right_right : List Nat := row9791_block_012 ++ row9791_block_013
theorem row9791_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9791 row9791_large_right_right_right :=
  Budget.RowUpperBound.branch row9791_block_012_covers row9791_block_013_covers

def row9791_large_right_right : List Nat := row9791_block_011 ++ row9791_large_right_right_right
theorem row9791_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9791 row9791_large_right_right :=
  Budget.RowUpperBound.branch row9791_block_011_covers row9791_large_right_right_right_covers

def row9791_large_right : List Nat := row9791_large_right_left ++ row9791_large_right_right
theorem row9791_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9791 row9791_large_right :=
  Budget.RowUpperBound.branch row9791_large_right_left_covers row9791_large_right_right_covers

def row9791_large : List Nat := row9791_large_left ++ row9791_large_right
theorem row9791_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9791 row9791_large :=
  Budget.RowUpperBound.branch row9791_large_left_covers row9791_large_right_covers

def row9791 : List Nat := row9791_large
theorem row9791_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9791 row9791 := row9791_large_covers

def row9803_small : List Nat := []
theorem row9803_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9803 row9803_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9803_block_005 : List Nat := [10061]
def plan9803_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9803_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9803 row9803_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9803_block_005
  · decide
  · rfl

def row9803_block_006 : List Nat := [11393, 11699]
def plan9803_block_006 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 31), (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 29), .second, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .second, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row9803_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9803 row9803_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9803_block_006
  · decide
  · rfl

def row9803_block_007 : List Nat := [12611, 12899, 13463]
def plan9803_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 29), (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), (.firstFactor 31), (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 29), (.secondFactor 19), .first, (.firstFactor 19), .second, .first, .second, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .possible, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9803_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9803 row9803_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9803_block_007
  · decide
  · rfl

def row9803_block_008 : List Nat := [14669, 16187]
def plan9803_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 23), .second, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 23), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 31), (.secondFactor 29), (.secondFactor 11), (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9803_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9803 row9803_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9803_block_008
  · decide
  · rfl

def row9803_block_009 : List Nat := []
theorem row9803_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9803 row9803_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9803_block_010 : List Nat := []
theorem row9803_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9803 row9803_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9803_block_011 : List Nat := []
theorem row9803_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9803 row9803_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9803_block_012 : List Nat := []
theorem row9803_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9803 row9803_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9803_block_013 : List Nat := []
theorem row9803_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9803 row9803_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9803_large_left_left : List Nat := row9803_small ++ row9803_block_005
theorem row9803_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9803 row9803_large_left_left :=
  Budget.RowUpperBound.branch row9803_small_covers row9803_block_005_covers

def row9803_large_left_right_right : List Nat := row9803_block_007 ++ row9803_block_008
theorem row9803_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9803 row9803_large_left_right_right :=
  Budget.RowUpperBound.branch row9803_block_007_covers row9803_block_008_covers

def row9803_large_left_right : List Nat := row9803_block_006 ++ row9803_large_left_right_right
theorem row9803_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9803 row9803_large_left_right :=
  Budget.RowUpperBound.branch row9803_block_006_covers row9803_large_left_right_right_covers

def row9803_large_left : List Nat := row9803_large_left_left ++ row9803_large_left_right
theorem row9803_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9803 row9803_large_left :=
  Budget.RowUpperBound.branch row9803_large_left_left_covers row9803_large_left_right_covers

def row9803_large_right_left : List Nat := row9803_block_009 ++ row9803_block_010
theorem row9803_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9803 row9803_large_right_left :=
  Budget.RowUpperBound.branch row9803_block_009_covers row9803_block_010_covers

def row9803_large_right_right_right : List Nat := row9803_block_012 ++ row9803_block_013
theorem row9803_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9803 row9803_large_right_right_right :=
  Budget.RowUpperBound.branch row9803_block_012_covers row9803_block_013_covers

def row9803_large_right_right : List Nat := row9803_block_011 ++ row9803_large_right_right_right
theorem row9803_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9803 row9803_large_right_right :=
  Budget.RowUpperBound.branch row9803_block_011_covers row9803_large_right_right_right_covers

def row9803_large_right : List Nat := row9803_large_right_left ++ row9803_large_right_right
theorem row9803_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9803 row9803_large_right :=
  Budget.RowUpperBound.branch row9803_large_right_left_covers row9803_large_right_right_covers

def row9803_large : List Nat := row9803_large_left ++ row9803_large_right
theorem row9803_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9803 row9803_large :=
  Budget.RowUpperBound.branch row9803_large_left_covers row9803_large_right_covers

def row9803 : List Nat := row9803_large
theorem row9803_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9803 row9803 := row9803_large_covers

def row9811_small : List Nat := []
theorem row9811_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9811 row9811_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9811_block_005 : List Nat := [10321]
def plan9811_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row9811_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9811 row9811_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9811_block_005
  · decide
  · rfl

def row9811_block_006 : List Nat := [10453, 11119]
def plan9811_block_006 : List Budget.Choice := [.first, .second, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9811_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9811 row9811_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9811_block_006
  · decide
  · rfl

def row9811_block_007 : List Nat := [12799, 13339, 13681, 13759]
def plan9811_block_007 : List Budget.Choice := [.second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 37), (.secondFactor 13), (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 19), .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9811_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9811 row9811_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9811_block_007
  · decide
  · rfl

def row9811_block_008 : List Nat := [15319, 15739, 16069]
def plan9811_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .second, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9811_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9811 row9811_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9811_block_008
  · decide
  · rfl

def row9811_block_009 : List Nat := []
theorem row9811_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9811 row9811_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9811_block_010 : List Nat := []
theorem row9811_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9811 row9811_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9811_block_011 : List Nat := []
theorem row9811_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9811 row9811_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9811_block_012 : List Nat := []
theorem row9811_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9811 row9811_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9811_block_013 : List Nat := []
theorem row9811_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9811 row9811_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9811_large_left_left : List Nat := row9811_small ++ row9811_block_005
theorem row9811_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9811 row9811_large_left_left :=
  Budget.RowUpperBound.branch row9811_small_covers row9811_block_005_covers

def row9811_large_left_right_right : List Nat := row9811_block_007 ++ row9811_block_008
theorem row9811_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9811 row9811_large_left_right_right :=
  Budget.RowUpperBound.branch row9811_block_007_covers row9811_block_008_covers

def row9811_large_left_right : List Nat := row9811_block_006 ++ row9811_large_left_right_right
theorem row9811_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9811 row9811_large_left_right :=
  Budget.RowUpperBound.branch row9811_block_006_covers row9811_large_left_right_right_covers

def row9811_large_left : List Nat := row9811_large_left_left ++ row9811_large_left_right
theorem row9811_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9811 row9811_large_left :=
  Budget.RowUpperBound.branch row9811_large_left_left_covers row9811_large_left_right_covers

def row9811_large_right_left : List Nat := row9811_block_009 ++ row9811_block_010
theorem row9811_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9811 row9811_large_right_left :=
  Budget.RowUpperBound.branch row9811_block_009_covers row9811_block_010_covers

def row9811_large_right_right_right : List Nat := row9811_block_012 ++ row9811_block_013
theorem row9811_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9811 row9811_large_right_right_right :=
  Budget.RowUpperBound.branch row9811_block_012_covers row9811_block_013_covers

def row9811_large_right_right : List Nat := row9811_block_011 ++ row9811_large_right_right_right
theorem row9811_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9811 row9811_large_right_right :=
  Budget.RowUpperBound.branch row9811_block_011_covers row9811_large_right_right_right_covers

def row9811_large_right : List Nat := row9811_large_right_left ++ row9811_large_right_right
theorem row9811_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9811 row9811_large_right :=
  Budget.RowUpperBound.branch row9811_large_right_left_covers row9811_large_right_right_covers

def row9811_large : List Nat := row9811_large_left ++ row9811_large_right
theorem row9811_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9811 row9811_large :=
  Budget.RowUpperBound.branch row9811_large_left_covers row9811_large_right_covers

def row9811 : List Nat := row9811_large
theorem row9811_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9811 row9811 := row9811_large_covers

end ProjectEulerSolutions.PrimePairLarge
