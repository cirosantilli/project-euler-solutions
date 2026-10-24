import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9851_small : List Nat := []
theorem row9851_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9851 row9851_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9851_block_005 : List Nat := [9857]
def plan9851_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9851_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9851 row9851_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9851_block_005
  · decide
  · rfl

def row9851_block_006 : List Nat := [11177, 11549, 11783, 12269]
def plan9851_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .possible, (.secondFactor 31), (.firstFactor 17), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .possible, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row9851_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9851 row9851_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9851_block_006
  · decide
  · rfl

def row9851_block_007 : List Nat := [13697]
def plan9851_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .second, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, .second, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .possible, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .second, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .second, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9851_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9851 row9851_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9851_block_007
  · decide
  · rfl

def row9851_block_008 : List Nat := [14699, 15233]
def plan9851_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 13), .second, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .possible, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9851_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9851 row9851_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9851_block_008
  · decide
  · rfl

def row9851_block_009 : List Nat := []
theorem row9851_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9851 row9851_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9851_block_010 : List Nat := []
theorem row9851_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9851 row9851_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9851_block_011 : List Nat := []
theorem row9851_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9851 row9851_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9851_block_012 : List Nat := []
theorem row9851_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9851 row9851_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9851_block_013 : List Nat := []
theorem row9851_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9851 row9851_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9851_large_left_left : List Nat := row9851_small ++ row9851_block_005
theorem row9851_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9851 row9851_large_left_left :=
  Budget.RowUpperBound.branch row9851_small_covers row9851_block_005_covers

def row9851_large_left_right_right : List Nat := row9851_block_007 ++ row9851_block_008
theorem row9851_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9851 row9851_large_left_right_right :=
  Budget.RowUpperBound.branch row9851_block_007_covers row9851_block_008_covers

def row9851_large_left_right : List Nat := row9851_block_006 ++ row9851_large_left_right_right
theorem row9851_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9851 row9851_large_left_right :=
  Budget.RowUpperBound.branch row9851_block_006_covers row9851_large_left_right_right_covers

def row9851_large_left : List Nat := row9851_large_left_left ++ row9851_large_left_right
theorem row9851_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9851 row9851_large_left :=
  Budget.RowUpperBound.branch row9851_large_left_left_covers row9851_large_left_right_covers

def row9851_large_right_left : List Nat := row9851_block_009 ++ row9851_block_010
theorem row9851_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9851 row9851_large_right_left :=
  Budget.RowUpperBound.branch row9851_block_009_covers row9851_block_010_covers

def row9851_large_right_right_right : List Nat := row9851_block_012 ++ row9851_block_013
theorem row9851_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9851 row9851_large_right_right_right :=
  Budget.RowUpperBound.branch row9851_block_012_covers row9851_block_013_covers

def row9851_large_right_right : List Nat := row9851_block_011 ++ row9851_large_right_right_right
theorem row9851_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9851 row9851_large_right_right :=
  Budget.RowUpperBound.branch row9851_block_011_covers row9851_large_right_right_right_covers

def row9851_large_right : List Nat := row9851_large_right_left ++ row9851_large_right_right
theorem row9851_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9851 row9851_large_right :=
  Budget.RowUpperBound.branch row9851_large_right_left_covers row9851_large_right_right_covers

def row9851_large : List Nat := row9851_large_left ++ row9851_large_right
theorem row9851_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9851 row9851_large :=
  Budget.RowUpperBound.branch row9851_large_left_covers row9851_large_right_covers

def row9851 : List Nat := row9851_large
theorem row9851_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9851 row9851 := row9851_large_covers

def row9857_small : List Nat := []
theorem row9857_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9857 row9857_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9857_block_005 : List Nat := [10181, 10331]
def plan9857_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .possible, (.firstFactor 13), (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 17), (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9857_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9857 row9857_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9857_block_005
  · decide
  · rfl

def row9857_block_006 : List Nat := [10391, 10433, 11057]
def plan9857_block_006 : List Budget.Choice := [.possible, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9857_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9857 row9857_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9857_block_006
  · decide
  · rfl

def row9857_block_007 : List Nat := [13241, 13679, 13997]
def plan9857_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .possible, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row9857_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9857 row9857_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9857_block_007
  · decide
  · rfl

def row9857_block_008 : List Nat := [14759, 15077]
def plan9857_block_008 : List Budget.Choice := [.first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 29), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9857_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9857 row9857_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9857_block_008
  · decide
  · rfl

def row9857_block_009 : List Nat := []
theorem row9857_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9857 row9857_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9857_block_010 : List Nat := []
theorem row9857_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9857 row9857_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9857_block_011 : List Nat := []
theorem row9857_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9857 row9857_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9857_block_012 : List Nat := []
theorem row9857_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9857 row9857_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9857_block_013 : List Nat := []
theorem row9857_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9857 row9857_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9857_large_left_left : List Nat := row9857_small ++ row9857_block_005
theorem row9857_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9857 row9857_large_left_left :=
  Budget.RowUpperBound.branch row9857_small_covers row9857_block_005_covers

def row9857_large_left_right_right : List Nat := row9857_block_007 ++ row9857_block_008
theorem row9857_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9857 row9857_large_left_right_right :=
  Budget.RowUpperBound.branch row9857_block_007_covers row9857_block_008_covers

def row9857_large_left_right : List Nat := row9857_block_006 ++ row9857_large_left_right_right
theorem row9857_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9857 row9857_large_left_right :=
  Budget.RowUpperBound.branch row9857_block_006_covers row9857_large_left_right_right_covers

def row9857_large_left : List Nat := row9857_large_left_left ++ row9857_large_left_right
theorem row9857_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9857 row9857_large_left :=
  Budget.RowUpperBound.branch row9857_large_left_left_covers row9857_large_left_right_covers

def row9857_large_right_left : List Nat := row9857_block_009 ++ row9857_block_010
theorem row9857_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9857 row9857_large_right_left :=
  Budget.RowUpperBound.branch row9857_block_009_covers row9857_block_010_covers

def row9857_large_right_right_right : List Nat := row9857_block_012 ++ row9857_block_013
theorem row9857_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9857 row9857_large_right_right_right :=
  Budget.RowUpperBound.branch row9857_block_012_covers row9857_block_013_covers

def row9857_large_right_right : List Nat := row9857_block_011 ++ row9857_large_right_right_right
theorem row9857_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9857 row9857_large_right_right :=
  Budget.RowUpperBound.branch row9857_block_011_covers row9857_large_right_right_right_covers

def row9857_large_right : List Nat := row9857_large_right_left ++ row9857_large_right_right
theorem row9857_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9857 row9857_large_right :=
  Budget.RowUpperBound.branch row9857_large_right_left_covers row9857_large_right_right_covers

def row9857_large : List Nat := row9857_large_left ++ row9857_large_right
theorem row9857_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9857 row9857_large :=
  Budget.RowUpperBound.branch row9857_large_left_covers row9857_large_right_covers

def row9857 : List Nat := row9857_large
theorem row9857_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9857 row9857 := row9857_large_covers

def row9859_small : List Nat := []
theorem row9859_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9859 row9859_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9859_block_005 : List Nat := [9967]
def plan9859_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), (.secondFactor 29), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row9859_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9859 row9859_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9859_block_005
  · decide
  · rfl

def row9859_block_006 : List Nat := [11497, 12241]
def plan9859_block_006 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .possible, (.secondFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9859_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9859 row9859_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9859_block_006
  · decide
  · rfl

def row9859_block_007 : List Nat := [12589, 13099]
def plan9859_block_007 : List Budget.Choice := [(.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .second, .possible, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9859_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9859 row9859_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9859_block_007
  · decide
  · rfl

def row9859_block_008 : List Nat := [15361]
def plan9859_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9859_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9859 row9859_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9859_block_008
  · decide
  · rfl

def row9859_block_009 : List Nat := []
theorem row9859_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9859 row9859_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9859_block_010 : List Nat := []
theorem row9859_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9859 row9859_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9859_block_011 : List Nat := []
theorem row9859_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9859 row9859_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9859_block_012 : List Nat := []
theorem row9859_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9859 row9859_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9859_block_013 : List Nat := []
theorem row9859_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9859 row9859_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9859_large_left_left : List Nat := row9859_small ++ row9859_block_005
theorem row9859_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9859 row9859_large_left_left :=
  Budget.RowUpperBound.branch row9859_small_covers row9859_block_005_covers

def row9859_large_left_right_right : List Nat := row9859_block_007 ++ row9859_block_008
theorem row9859_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9859 row9859_large_left_right_right :=
  Budget.RowUpperBound.branch row9859_block_007_covers row9859_block_008_covers

def row9859_large_left_right : List Nat := row9859_block_006 ++ row9859_large_left_right_right
theorem row9859_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9859 row9859_large_left_right :=
  Budget.RowUpperBound.branch row9859_block_006_covers row9859_large_left_right_right_covers

def row9859_large_left : List Nat := row9859_large_left_left ++ row9859_large_left_right
theorem row9859_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9859 row9859_large_left :=
  Budget.RowUpperBound.branch row9859_large_left_left_covers row9859_large_left_right_covers

def row9859_large_right_left : List Nat := row9859_block_009 ++ row9859_block_010
theorem row9859_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9859 row9859_large_right_left :=
  Budget.RowUpperBound.branch row9859_block_009_covers row9859_block_010_covers

def row9859_large_right_right_right : List Nat := row9859_block_012 ++ row9859_block_013
theorem row9859_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9859 row9859_large_right_right_right :=
  Budget.RowUpperBound.branch row9859_block_012_covers row9859_block_013_covers

def row9859_large_right_right : List Nat := row9859_block_011 ++ row9859_large_right_right_right
theorem row9859_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9859 row9859_large_right_right :=
  Budget.RowUpperBound.branch row9859_block_011_covers row9859_large_right_right_right_covers

def row9859_large_right : List Nat := row9859_large_right_left ++ row9859_large_right_right
theorem row9859_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9859 row9859_large_right :=
  Budget.RowUpperBound.branch row9859_large_right_left_covers row9859_large_right_right_covers

def row9859_large : List Nat := row9859_large_left ++ row9859_large_right
theorem row9859_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9859 row9859_large :=
  Budget.RowUpperBound.branch row9859_large_left_covers row9859_large_right_covers

def row9859 : List Nat := row9859_large
theorem row9859_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9859 row9859 := row9859_large_covers

def row9871_small : List Nat := []
theorem row9871_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9871 row9871_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9871_block_005 : List Nat := [10177]
def plan9871_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9871_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9871 row9871_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9871_block_005
  · decide
  · rfl

def row9871_block_006 : List Nat := [11497, 11551]
def plan9871_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 23), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .second, .first, .first, .first, (.secondFactor 11), .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 29), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9871_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9871 row9871_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9871_block_006
  · decide
  · rfl

def row9871_block_007 : List Nat := [13063, 13597, 14281]
def plan9871_block_007 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 23), .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .possible, .second, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 13), .possible, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .second, .first, (.firstFactor 31), (.firstFactor 23), .first, .second, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9871_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9871 row9871_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9871_block_007
  · decide
  · rfl

def row9871_block_008 : List Nat := [15373, 16069]
def plan9871_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 19), .first, .possible, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9871_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9871 row9871_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9871_block_008
  · decide
  · rfl

def row9871_block_009 : List Nat := []
theorem row9871_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9871 row9871_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9871_block_010 : List Nat := []
theorem row9871_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9871 row9871_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9871_block_011 : List Nat := []
theorem row9871_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9871 row9871_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9871_block_012 : List Nat := []
theorem row9871_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9871 row9871_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9871_block_013 : List Nat := []
theorem row9871_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9871 row9871_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9871_large_left_left : List Nat := row9871_small ++ row9871_block_005
theorem row9871_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9871 row9871_large_left_left :=
  Budget.RowUpperBound.branch row9871_small_covers row9871_block_005_covers

def row9871_large_left_right_right : List Nat := row9871_block_007 ++ row9871_block_008
theorem row9871_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9871 row9871_large_left_right_right :=
  Budget.RowUpperBound.branch row9871_block_007_covers row9871_block_008_covers

def row9871_large_left_right : List Nat := row9871_block_006 ++ row9871_large_left_right_right
theorem row9871_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9871 row9871_large_left_right :=
  Budget.RowUpperBound.branch row9871_block_006_covers row9871_large_left_right_right_covers

def row9871_large_left : List Nat := row9871_large_left_left ++ row9871_large_left_right
theorem row9871_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9871 row9871_large_left :=
  Budget.RowUpperBound.branch row9871_large_left_left_covers row9871_large_left_right_covers

def row9871_large_right_left : List Nat := row9871_block_009 ++ row9871_block_010
theorem row9871_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9871 row9871_large_right_left :=
  Budget.RowUpperBound.branch row9871_block_009_covers row9871_block_010_covers

def row9871_large_right_right_right : List Nat := row9871_block_012 ++ row9871_block_013
theorem row9871_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9871 row9871_large_right_right_right :=
  Budget.RowUpperBound.branch row9871_block_012_covers row9871_block_013_covers

def row9871_large_right_right : List Nat := row9871_block_011 ++ row9871_large_right_right_right
theorem row9871_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9871 row9871_large_right_right :=
  Budget.RowUpperBound.branch row9871_block_011_covers row9871_large_right_right_right_covers

def row9871_large_right : List Nat := row9871_large_right_left ++ row9871_large_right_right
theorem row9871_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9871 row9871_large_right :=
  Budget.RowUpperBound.branch row9871_large_right_left_covers row9871_large_right_right_covers

def row9871_large : List Nat := row9871_large_left ++ row9871_large_right
theorem row9871_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9871 row9871_large :=
  Budget.RowUpperBound.branch row9871_large_left_covers row9871_large_right_covers

def row9871 : List Nat := row9871_large
theorem row9871_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9871 row9871 := row9871_large_covers

end ProjectEulerSolutions.PrimePairLarge
