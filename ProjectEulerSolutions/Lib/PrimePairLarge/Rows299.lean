import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9721_small : List Nat := []
theorem row9721_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9721 row9721_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9721_block_005 : List Nat := [9901]
def plan9721_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row9721_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9721 row9721_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9721_block_005
  · decide
  · rfl

def row9721_block_006 : List Nat := [10957, 11251, 11551, 11587, 11689]
def plan9721_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .second, .first, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .possible, .first, .possible, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .second, .first, .second, .first, .first, .first, .second, .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row9721_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9721 row9721_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9721_block_006
  · decide
  · rfl

def row9721_block_007 : List Nat := [13033, 13063, 13999]
def plan9721_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 29), (.firstFactor 23), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 11), (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9721_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9721 row9721_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9721_block_007
  · decide
  · rfl

def row9721_block_008 : List Nat := [14629, 14929, 15013, 15121, 15427, 15889, 16267]
def plan9721_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, (.firstFactor 29), (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .second, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 37), .first, (.secondFactor 19), .first, .first, (.firstFactor 29), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9721_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9721 row9721_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9721_block_008
  · decide
  · rfl

def row9721_block_009 : List Nat := []
theorem row9721_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9721 row9721_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9721_block_010 : List Nat := []
theorem row9721_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9721 row9721_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9721_block_011 : List Nat := []
theorem row9721_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9721 row9721_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9721_block_012 : List Nat := []
theorem row9721_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9721 row9721_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9721_block_013 : List Nat := []
theorem row9721_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9721 row9721_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9721_large_left_left : List Nat := row9721_small ++ row9721_block_005
theorem row9721_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9721 row9721_large_left_left :=
  Budget.RowUpperBound.branch row9721_small_covers row9721_block_005_covers

def row9721_large_left_right_right : List Nat := row9721_block_007 ++ row9721_block_008
theorem row9721_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9721 row9721_large_left_right_right :=
  Budget.RowUpperBound.branch row9721_block_007_covers row9721_block_008_covers

def row9721_large_left_right : List Nat := row9721_block_006 ++ row9721_large_left_right_right
theorem row9721_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9721 row9721_large_left_right :=
  Budget.RowUpperBound.branch row9721_block_006_covers row9721_large_left_right_right_covers

def row9721_large_left : List Nat := row9721_large_left_left ++ row9721_large_left_right
theorem row9721_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9721 row9721_large_left :=
  Budget.RowUpperBound.branch row9721_large_left_left_covers row9721_large_left_right_covers

def row9721_large_right_left : List Nat := row9721_block_009 ++ row9721_block_010
theorem row9721_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9721 row9721_large_right_left :=
  Budget.RowUpperBound.branch row9721_block_009_covers row9721_block_010_covers

def row9721_large_right_right_right : List Nat := row9721_block_012 ++ row9721_block_013
theorem row9721_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9721 row9721_large_right_right_right :=
  Budget.RowUpperBound.branch row9721_block_012_covers row9721_block_013_covers

def row9721_large_right_right : List Nat := row9721_block_011 ++ row9721_large_right_right_right
theorem row9721_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9721 row9721_large_right_right :=
  Budget.RowUpperBound.branch row9721_block_011_covers row9721_large_right_right_right_covers

def row9721_large_right : List Nat := row9721_large_right_left ++ row9721_large_right_right
theorem row9721_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9721 row9721_large_right :=
  Budget.RowUpperBound.branch row9721_large_right_left_covers row9721_large_right_right_covers

def row9721_large : List Nat := row9721_large_left ++ row9721_large_right
theorem row9721_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9721 row9721_large :=
  Budget.RowUpperBound.branch row9721_large_left_covers row9721_large_right_covers

def row9721 : List Nat := row9721_large
theorem row9721_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9721 row9721 := row9721_large_covers

def row9733_small : List Nat := []
theorem row9733_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9733 row9733_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9733_block_005 : List Nat := [9769, 9817, 9883, 10369]
def plan9733_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .possible, .first, (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .possible]
set_option maxRecDepth 2048 in
theorem row9733_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9733 row9733_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9733_block_005
  · decide
  · rfl

def row9733_block_006 : List Nat := [10957, 11113, 12037]
def plan9733_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 11), .second, .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9733_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9733 row9733_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9733_block_006
  · decide
  · rfl

def row9733_block_007 : List Nat := [12583, 13591, 13681]
def plan9733_block_007 : List Budget.Choice := [(.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 11), .second, .first, .first, .first, .possible, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 31), .first, .first, .second, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9733_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9733 row9733_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9733_block_007
  · decide
  · rfl

def row9733_block_008 : List Nat := [15559, 15619, 15991]
def plan9733_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 7), .first, .second, (.firstFactor 29), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .possible, .first, .first, (.secondFactor 31), .first, (.firstFactor 31), (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9733_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9733 row9733_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9733_block_008
  · decide
  · rfl

def row9733_block_009 : List Nat := []
theorem row9733_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9733 row9733_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9733_block_010 : List Nat := []
theorem row9733_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9733 row9733_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9733_block_011 : List Nat := []
theorem row9733_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9733 row9733_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9733_block_012 : List Nat := []
theorem row9733_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9733 row9733_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9733_block_013 : List Nat := []
theorem row9733_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9733 row9733_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9733_large_left_left : List Nat := row9733_small ++ row9733_block_005
theorem row9733_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9733 row9733_large_left_left :=
  Budget.RowUpperBound.branch row9733_small_covers row9733_block_005_covers

def row9733_large_left_right_right : List Nat := row9733_block_007 ++ row9733_block_008
theorem row9733_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9733 row9733_large_left_right_right :=
  Budget.RowUpperBound.branch row9733_block_007_covers row9733_block_008_covers

def row9733_large_left_right : List Nat := row9733_block_006 ++ row9733_large_left_right_right
theorem row9733_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9733 row9733_large_left_right :=
  Budget.RowUpperBound.branch row9733_block_006_covers row9733_large_left_right_right_covers

def row9733_large_left : List Nat := row9733_large_left_left ++ row9733_large_left_right
theorem row9733_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9733 row9733_large_left :=
  Budget.RowUpperBound.branch row9733_large_left_left_covers row9733_large_left_right_covers

def row9733_large_right_left : List Nat := row9733_block_009 ++ row9733_block_010
theorem row9733_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9733 row9733_large_right_left :=
  Budget.RowUpperBound.branch row9733_block_009_covers row9733_block_010_covers

def row9733_large_right_right_right : List Nat := row9733_block_012 ++ row9733_block_013
theorem row9733_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9733 row9733_large_right_right_right :=
  Budget.RowUpperBound.branch row9733_block_012_covers row9733_block_013_covers

def row9733_large_right_right : List Nat := row9733_block_011 ++ row9733_large_right_right_right
theorem row9733_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9733 row9733_large_right_right :=
  Budget.RowUpperBound.branch row9733_block_011_covers row9733_large_right_right_right_covers

def row9733_large_right : List Nat := row9733_large_right_left ++ row9733_large_right_right
theorem row9733_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9733 row9733_large_right :=
  Budget.RowUpperBound.branch row9733_large_right_left_covers row9733_large_right_right_covers

def row9733_large : List Nat := row9733_large_left ++ row9733_large_right
theorem row9733_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9733 row9733_large :=
  Budget.RowUpperBound.branch row9733_large_left_covers row9733_large_right_covers

def row9733 : List Nat := row9733_large
theorem row9733_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9733 row9733 := row9733_large_covers

def row9739_small : List Nat := []
theorem row9739_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9739 row9739_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9739_block_005 : List Nat := [9769, 9781, 10159]
def plan9739_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .possible, .possible, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row9739_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9739 row9739_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9739_block_005
  · decide
  · rfl

def row9739_block_006 : List Nat := [11353, 11923, 12343]
def plan9739_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .second, .first, (.secondFactor 23), .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row9739_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9739 row9739_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9739_block_006
  · decide
  · rfl

def row9739_block_007 : List Nat := [13033, 14083, 14341]
def plan9739_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 23), (.secondFactor 31), .first, (.secondFactor 7), (.secondFactor 19), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row9739_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9739 row9739_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9739_block_007
  · decide
  · rfl

def row9739_block_008 : List Nat := [15031, 15331, 15559, 15919, 15937]
def plan9739_block_008 : List Budget.Choice := [(.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .possible, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, .second, .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .possible, .first, .possible, .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9739_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9739 row9739_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9739_block_008
  · decide
  · rfl

def row9739_block_009 : List Nat := []
theorem row9739_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9739 row9739_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9739_block_010 : List Nat := []
theorem row9739_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9739 row9739_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9739_block_011 : List Nat := []
theorem row9739_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9739 row9739_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9739_block_012 : List Nat := []
theorem row9739_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9739 row9739_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9739_block_013 : List Nat := []
theorem row9739_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9739 row9739_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9739_large_left_left : List Nat := row9739_small ++ row9739_block_005
theorem row9739_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9739 row9739_large_left_left :=
  Budget.RowUpperBound.branch row9739_small_covers row9739_block_005_covers

def row9739_large_left_right_right : List Nat := row9739_block_007 ++ row9739_block_008
theorem row9739_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9739 row9739_large_left_right_right :=
  Budget.RowUpperBound.branch row9739_block_007_covers row9739_block_008_covers

def row9739_large_left_right : List Nat := row9739_block_006 ++ row9739_large_left_right_right
theorem row9739_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9739 row9739_large_left_right :=
  Budget.RowUpperBound.branch row9739_block_006_covers row9739_large_left_right_right_covers

def row9739_large_left : List Nat := row9739_large_left_left ++ row9739_large_left_right
theorem row9739_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9739 row9739_large_left :=
  Budget.RowUpperBound.branch row9739_large_left_left_covers row9739_large_left_right_covers

def row9739_large_right_left : List Nat := row9739_block_009 ++ row9739_block_010
theorem row9739_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9739 row9739_large_right_left :=
  Budget.RowUpperBound.branch row9739_block_009_covers row9739_block_010_covers

def row9739_large_right_right_right : List Nat := row9739_block_012 ++ row9739_block_013
theorem row9739_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9739 row9739_large_right_right_right :=
  Budget.RowUpperBound.branch row9739_block_012_covers row9739_block_013_covers

def row9739_large_right_right : List Nat := row9739_block_011 ++ row9739_large_right_right_right
theorem row9739_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9739 row9739_large_right_right :=
  Budget.RowUpperBound.branch row9739_block_011_covers row9739_large_right_right_right_covers

def row9739_large_right : List Nat := row9739_large_right_left ++ row9739_large_right_right
theorem row9739_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9739 row9739_large_right :=
  Budget.RowUpperBound.branch row9739_large_right_left_covers row9739_large_right_right_covers

def row9739_large : List Nat := row9739_large_left ++ row9739_large_right
theorem row9739_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9739 row9739_large :=
  Budget.RowUpperBound.branch row9739_large_left_covers row9739_large_right_covers

def row9739 : List Nat := row9739_large
theorem row9739_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9739 row9739 := row9739_large_covers

def row9743_small : List Nat := []
theorem row9743_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9743 row9743_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9743_block_005 : List Nat := []
def plan9743_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 29), .second, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row9743_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9743 row9743_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9743_block_005
  · decide
  · rfl

def row9743_block_006 : List Nat := [10949, 10979, 11969]
def plan9743_block_006 : List Budget.Choice := [.first, .first, .second, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 37), .possible, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .possible, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9743_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9743 row9743_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9743_block_006
  · decide
  · rfl

def row9743_block_007 : List Nat := [12611, 12743, 13259]
def plan9743_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 11), .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 29), (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9743_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9743 row9743_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9743_block_007
  · decide
  · rfl

def row9743_block_008 : List Nat := [14669, 15959, 16253]
def plan9743_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, (.firstFactor 37), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .possible, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9743_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9743 row9743_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9743_block_008
  · decide
  · rfl

def row9743_block_009 : List Nat := []
theorem row9743_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9743 row9743_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9743_block_010 : List Nat := []
theorem row9743_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9743 row9743_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9743_block_011 : List Nat := []
theorem row9743_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9743 row9743_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9743_block_012 : List Nat := []
theorem row9743_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9743 row9743_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9743_block_013 : List Nat := []
theorem row9743_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9743 row9743_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9743_large_left_left : List Nat := row9743_small ++ row9743_block_005
theorem row9743_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9743 row9743_large_left_left :=
  Budget.RowUpperBound.branch row9743_small_covers row9743_block_005_covers

def row9743_large_left_right_right : List Nat := row9743_block_007 ++ row9743_block_008
theorem row9743_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9743 row9743_large_left_right_right :=
  Budget.RowUpperBound.branch row9743_block_007_covers row9743_block_008_covers

def row9743_large_left_right : List Nat := row9743_block_006 ++ row9743_large_left_right_right
theorem row9743_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9743 row9743_large_left_right :=
  Budget.RowUpperBound.branch row9743_block_006_covers row9743_large_left_right_right_covers

def row9743_large_left : List Nat := row9743_large_left_left ++ row9743_large_left_right
theorem row9743_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9743 row9743_large_left :=
  Budget.RowUpperBound.branch row9743_large_left_left_covers row9743_large_left_right_covers

def row9743_large_right_left : List Nat := row9743_block_009 ++ row9743_block_010
theorem row9743_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9743 row9743_large_right_left :=
  Budget.RowUpperBound.branch row9743_block_009_covers row9743_block_010_covers

def row9743_large_right_right_right : List Nat := row9743_block_012 ++ row9743_block_013
theorem row9743_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9743 row9743_large_right_right_right :=
  Budget.RowUpperBound.branch row9743_block_012_covers row9743_block_013_covers

def row9743_large_right_right : List Nat := row9743_block_011 ++ row9743_large_right_right_right
theorem row9743_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9743 row9743_large_right_right :=
  Budget.RowUpperBound.branch row9743_block_011_covers row9743_large_right_right_right_covers

def row9743_large_right : List Nat := row9743_large_right_left ++ row9743_large_right_right
theorem row9743_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9743 row9743_large_right :=
  Budget.RowUpperBound.branch row9743_large_right_left_covers row9743_large_right_right_covers

def row9743_large : List Nat := row9743_large_left ++ row9743_large_right
theorem row9743_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9743 row9743_large :=
  Budget.RowUpperBound.branch row9743_large_left_covers row9743_large_right_covers

def row9743 : List Nat := row9743_large
theorem row9743_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9743 row9743 := row9743_large_covers

end ProjectEulerSolutions.PrimePairLarge
