import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9067_small : List Nat := []
theorem row9067_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9067 row9067_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9067_block_005 : List Nat := [9349, 9769, 9811, 10369]
def plan9067_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 19), .second, .first, .first, .second, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 23), .first, (.secondFactor 37), .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .first, .possible, (.firstFactor 29), (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .possible]
set_option maxRecDepth 2048 in
theorem row9067_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9067 row9067_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9067_block_005
  · decide
  · rfl

def row9067_block_006 : List Nat := [10501, 10687, 10891, 11551]
def plan9067_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .second, .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 17), .second, .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, .second, .first, .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .second, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row9067_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9067 row9067_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9067_block_006
  · decide
  · rfl

def row9067_block_007 : List Nat := [13477, 13627, 14173]
def plan9067_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 29), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .second]
set_option maxRecDepth 2048 in
theorem row9067_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9067 row9067_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9067_block_007
  · decide
  · rfl

def row9067_block_008 : List Nat := [14503, 15289, 15619, 15787, 16249]
def plan9067_block_008 : List Budget.Choice := [(.firstFactor 11), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 19), (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 13), .second, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9067_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9067 row9067_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9067_block_008
  · decide
  · rfl

def row9067_block_009 : List Nat := []
def plan9067_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9067_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9067 row9067_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9067_block_009
  · decide
  · rfl

def row9067_block_010 : List Nat := []
theorem row9067_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9067 row9067_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9067_block_011 : List Nat := []
theorem row9067_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9067 row9067_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9067_block_012 : List Nat := []
theorem row9067_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9067 row9067_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9067_block_013 : List Nat := []
theorem row9067_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9067 row9067_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9067_large_left_left : List Nat := row9067_small ++ row9067_block_005
theorem row9067_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9067 row9067_large_left_left :=
  Budget.RowUpperBound.branch row9067_small_covers row9067_block_005_covers

def row9067_large_left_right_right : List Nat := row9067_block_007 ++ row9067_block_008
theorem row9067_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9067 row9067_large_left_right_right :=
  Budget.RowUpperBound.branch row9067_block_007_covers row9067_block_008_covers

def row9067_large_left_right : List Nat := row9067_block_006 ++ row9067_large_left_right_right
theorem row9067_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9067 row9067_large_left_right :=
  Budget.RowUpperBound.branch row9067_block_006_covers row9067_large_left_right_right_covers

def row9067_large_left : List Nat := row9067_large_left_left ++ row9067_large_left_right
theorem row9067_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9067 row9067_large_left :=
  Budget.RowUpperBound.branch row9067_large_left_left_covers row9067_large_left_right_covers

def row9067_large_right_left : List Nat := row9067_block_009 ++ row9067_block_010
theorem row9067_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9067 row9067_large_right_left :=
  Budget.RowUpperBound.branch row9067_block_009_covers row9067_block_010_covers

def row9067_large_right_right_right : List Nat := row9067_block_012 ++ row9067_block_013
theorem row9067_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9067 row9067_large_right_right_right :=
  Budget.RowUpperBound.branch row9067_block_012_covers row9067_block_013_covers

def row9067_large_right_right : List Nat := row9067_block_011 ++ row9067_large_right_right_right
theorem row9067_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9067 row9067_large_right_right :=
  Budget.RowUpperBound.branch row9067_block_011_covers row9067_large_right_right_right_covers

def row9067_large_right : List Nat := row9067_large_right_left ++ row9067_large_right_right
theorem row9067_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9067 row9067_large_right :=
  Budget.RowUpperBound.branch row9067_large_right_left_covers row9067_large_right_right_covers

def row9067_large : List Nat := row9067_large_left ++ row9067_large_right
theorem row9067_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9067 row9067_large :=
  Budget.RowUpperBound.branch row9067_large_left_covers row9067_large_right_covers

def row9067 : List Nat := row9067_large
theorem row9067_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9067 row9067 := row9067_large_covers

def row9091_small : List Nat := []
theorem row9091_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9091 row9091_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9091_block_005 : List Nat := [9511, 9733, 9781, 10333]
def plan9091_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.firstFactor 37), .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 23), .possible, .first, .first, .first, .first, (.firstFactor 19), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row9091_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9091 row9091_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9091_block_005
  · decide
  · rfl

def row9091_block_006 : List Nat := [10597, 10639, 10957, 11251, 11299]
def plan9091_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 23), .first, .possible, (.firstFactor 19), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .second, (.secondFactor 31), .first, .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 37), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row9091_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9091 row9091_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9091_block_006
  · decide
  · rfl

def row9091_block_007 : List Nat := [12433, 12907, 12979, 13963]
def plan9091_block_007 : List Budget.Choice := [(.firstFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 17), .first, .second, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 29), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9091_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9091 row9091_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9091_block_007
  · decide
  · rfl

def row9091_block_008 : List Nat := [14737, 16189, 16333, 16339]
def plan9091_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 31), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .second, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .second, .first, .second, (.firstFactor 13), .first, .first, .possible, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9091_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9091 row9091_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9091_block_008
  · decide
  · rfl

def row9091_block_009 : List Nat := []
def plan9091_block_009 : List Budget.Choice := [(.secondFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9091_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9091 row9091_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9091_block_009
  · decide
  · rfl

def row9091_block_010 : List Nat := []
theorem row9091_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9091 row9091_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9091_block_011 : List Nat := []
theorem row9091_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9091 row9091_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9091_block_012 : List Nat := []
theorem row9091_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9091 row9091_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9091_block_013 : List Nat := []
theorem row9091_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9091 row9091_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9091_large_left_left : List Nat := row9091_small ++ row9091_block_005
theorem row9091_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9091 row9091_large_left_left :=
  Budget.RowUpperBound.branch row9091_small_covers row9091_block_005_covers

def row9091_large_left_right_right : List Nat := row9091_block_007 ++ row9091_block_008
theorem row9091_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9091 row9091_large_left_right_right :=
  Budget.RowUpperBound.branch row9091_block_007_covers row9091_block_008_covers

def row9091_large_left_right : List Nat := row9091_block_006 ++ row9091_large_left_right_right
theorem row9091_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9091 row9091_large_left_right :=
  Budget.RowUpperBound.branch row9091_block_006_covers row9091_large_left_right_right_covers

def row9091_large_left : List Nat := row9091_large_left_left ++ row9091_large_left_right
theorem row9091_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9091 row9091_large_left :=
  Budget.RowUpperBound.branch row9091_large_left_left_covers row9091_large_left_right_covers

def row9091_large_right_left : List Nat := row9091_block_009 ++ row9091_block_010
theorem row9091_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9091 row9091_large_right_left :=
  Budget.RowUpperBound.branch row9091_block_009_covers row9091_block_010_covers

def row9091_large_right_right_right : List Nat := row9091_block_012 ++ row9091_block_013
theorem row9091_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9091 row9091_large_right_right_right :=
  Budget.RowUpperBound.branch row9091_block_012_covers row9091_block_013_covers

def row9091_large_right_right : List Nat := row9091_block_011 ++ row9091_large_right_right_right
theorem row9091_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9091 row9091_large_right_right :=
  Budget.RowUpperBound.branch row9091_block_011_covers row9091_large_right_right_right_covers

def row9091_large_right : List Nat := row9091_large_right_left ++ row9091_large_right_right
theorem row9091_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9091 row9091_large_right :=
  Budget.RowUpperBound.branch row9091_large_right_left_covers row9091_large_right_right_covers

def row9091_large : List Nat := row9091_large_left ++ row9091_large_right
theorem row9091_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9091 row9091_large :=
  Budget.RowUpperBound.branch row9091_large_left_covers row9091_large_right_covers

def row9091 : List Nat := row9091_large
theorem row9091_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9091 row9091 := row9091_large_covers

def row9103_small : List Nat := []
theorem row9103_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9103 row9103_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9103_block_005 : List Nat := [9181, 9649, 10069, 10369]
def plan9103_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, (.secondFactor 37), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 31), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 37), .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .possible]
set_option maxRecDepth 2048 in
theorem row9103_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9103 row9103_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9103_block_005
  · decide
  · rfl

def row9103_block_006 : List Nat := [10651, 10909, 11317, 11953]
def plan9103_block_006 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, (.secondFactor 17), .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 29), .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9103_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9103 row9103_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9103_block_006
  · decide
  · rfl

def row9103_block_007 : List Nat := [12541]
def plan9103_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, .first, .possible, (.secondFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 11), (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 31), (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 19), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9103_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9103 row9103_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9103_block_007
  · decide
  · rfl

def row9103_block_008 : List Nat := [14653, 14983, 15277, 15733]
def plan9103_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 13), .second, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9103_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9103 row9103_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9103_block_008
  · decide
  · rfl

def row9103_block_009 : List Nat := []
def plan9103_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9103_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9103 row9103_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9103_block_009
  · decide
  · rfl

def row9103_block_010 : List Nat := []
theorem row9103_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9103 row9103_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9103_block_011 : List Nat := []
theorem row9103_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9103 row9103_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9103_block_012 : List Nat := []
theorem row9103_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9103 row9103_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9103_block_013 : List Nat := []
theorem row9103_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9103 row9103_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9103_large_left_left : List Nat := row9103_small ++ row9103_block_005
theorem row9103_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9103 row9103_large_left_left :=
  Budget.RowUpperBound.branch row9103_small_covers row9103_block_005_covers

def row9103_large_left_right_right : List Nat := row9103_block_007 ++ row9103_block_008
theorem row9103_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9103 row9103_large_left_right_right :=
  Budget.RowUpperBound.branch row9103_block_007_covers row9103_block_008_covers

def row9103_large_left_right : List Nat := row9103_block_006 ++ row9103_large_left_right_right
theorem row9103_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9103 row9103_large_left_right :=
  Budget.RowUpperBound.branch row9103_block_006_covers row9103_large_left_right_right_covers

def row9103_large_left : List Nat := row9103_large_left_left ++ row9103_large_left_right
theorem row9103_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9103 row9103_large_left :=
  Budget.RowUpperBound.branch row9103_large_left_left_covers row9103_large_left_right_covers

def row9103_large_right_left : List Nat := row9103_block_009 ++ row9103_block_010
theorem row9103_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9103 row9103_large_right_left :=
  Budget.RowUpperBound.branch row9103_block_009_covers row9103_block_010_covers

def row9103_large_right_right_right : List Nat := row9103_block_012 ++ row9103_block_013
theorem row9103_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9103 row9103_large_right_right_right :=
  Budget.RowUpperBound.branch row9103_block_012_covers row9103_block_013_covers

def row9103_large_right_right : List Nat := row9103_block_011 ++ row9103_large_right_right_right
theorem row9103_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9103 row9103_large_right_right :=
  Budget.RowUpperBound.branch row9103_block_011_covers row9103_large_right_right_right_covers

def row9103_large_right : List Nat := row9103_large_right_left ++ row9103_large_right_right
theorem row9103_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9103 row9103_large_right :=
  Budget.RowUpperBound.branch row9103_large_right_left_covers row9103_large_right_right_covers

def row9103_large : List Nat := row9103_large_left ++ row9103_large_right
theorem row9103_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9103 row9103_large :=
  Budget.RowUpperBound.branch row9103_large_left_covers row9103_large_right_covers

def row9103 : List Nat := row9103_large
theorem row9103_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9103 row9103 := row9103_large_covers

def row9109_small : List Nat := []
theorem row9109_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9109 row9109_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9109_block_005 : List Nat := [9133, 9181, 9871, 9949]
def plan9109_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .possible, .first, (.firstFactor 11), .second, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9109_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9109 row9109_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9109_block_005
  · decide
  · rfl

def row9109_block_006 : List Nat := [10723, 10771, 11467]
def plan9109_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .possible, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .second, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row9109_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9109 row9109_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9109_block_006
  · decide
  · rfl

def row9109_block_007 : List Nat := [12553]
def plan9109_block_007 : List Budget.Choice := [(.firstFactor 19), (.secondFactor 31), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .second, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9109_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9109 row9109_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9109_block_007
  · decide
  · rfl

def row9109_block_008 : List Nat := [15073, 15121, 15349, 15919, 16249, 16381]
def plan9109_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 11), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .second, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .possible, .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 19), .possible, .first, .first, (.secondFactor 13), .first, .first, .second, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9109_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9109 row9109_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9109_block_008
  · decide
  · rfl

def row9109_block_009 : List Nat := []
def plan9109_block_009 : List Budget.Choice := [(.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9109_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9109 row9109_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9109_block_009
  · decide
  · rfl

def row9109_block_010 : List Nat := []
theorem row9109_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9109 row9109_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9109_block_011 : List Nat := []
theorem row9109_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9109 row9109_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9109_block_012 : List Nat := []
theorem row9109_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9109 row9109_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9109_block_013 : List Nat := []
theorem row9109_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9109 row9109_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9109_large_left_left : List Nat := row9109_small ++ row9109_block_005
theorem row9109_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9109 row9109_large_left_left :=
  Budget.RowUpperBound.branch row9109_small_covers row9109_block_005_covers

def row9109_large_left_right_right : List Nat := row9109_block_007 ++ row9109_block_008
theorem row9109_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9109 row9109_large_left_right_right :=
  Budget.RowUpperBound.branch row9109_block_007_covers row9109_block_008_covers

def row9109_large_left_right : List Nat := row9109_block_006 ++ row9109_large_left_right_right
theorem row9109_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9109 row9109_large_left_right :=
  Budget.RowUpperBound.branch row9109_block_006_covers row9109_large_left_right_right_covers

def row9109_large_left : List Nat := row9109_large_left_left ++ row9109_large_left_right
theorem row9109_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9109 row9109_large_left :=
  Budget.RowUpperBound.branch row9109_large_left_left_covers row9109_large_left_right_covers

def row9109_large_right_left : List Nat := row9109_block_009 ++ row9109_block_010
theorem row9109_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9109 row9109_large_right_left :=
  Budget.RowUpperBound.branch row9109_block_009_covers row9109_block_010_covers

def row9109_large_right_right_right : List Nat := row9109_block_012 ++ row9109_block_013
theorem row9109_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9109 row9109_large_right_right_right :=
  Budget.RowUpperBound.branch row9109_block_012_covers row9109_block_013_covers

def row9109_large_right_right : List Nat := row9109_block_011 ++ row9109_large_right_right_right
theorem row9109_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9109 row9109_large_right_right :=
  Budget.RowUpperBound.branch row9109_block_011_covers row9109_large_right_right_right_covers

def row9109_large_right : List Nat := row9109_large_right_left ++ row9109_large_right_right
theorem row9109_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9109 row9109_large_right :=
  Budget.RowUpperBound.branch row9109_large_right_left_covers row9109_large_right_right_covers

def row9109_large : List Nat := row9109_large_left ++ row9109_large_right
theorem row9109_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9109 row9109_large :=
  Budget.RowUpperBound.branch row9109_large_left_covers row9109_large_right_covers

def row9109 : List Nat := row9109_large
theorem row9109_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9109 row9109 := row9109_large_covers

end ProjectEulerSolutions.PrimePairLarge
