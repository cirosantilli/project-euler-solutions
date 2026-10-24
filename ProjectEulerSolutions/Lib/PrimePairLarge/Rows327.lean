import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10753_small : List Nat := []
theorem row10753_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10753 row10753_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10753_block_005 : List Nat := []
theorem row10753_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10753 row10753_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10753_block_006 : List Nat := [10993, 11119, 11437, 12409]
def plan10753_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 37), (.firstFactor 11), .first, .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, (.firstFactor 19), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row10753_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10753 row10753_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10753_block_006
  · decide
  · rfl

def row10753_block_007 : List Nat := [12829, 13033, 13441, 14437]
def plan10753_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .possible, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .possible, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 23), .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 19), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .second, (.secondFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .second, .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10753_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10753 row10753_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10753_block_007
  · decide
  · rfl

def row10753_block_008 : List Nat := [14629, 14731]
def plan10753_block_008 : List Budget.Choice := [(.secondFactor 29), .first, .first, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 17), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 7), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10753_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10753 row10753_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10753_block_008
  · decide
  · rfl

def row10753_block_009 : List Nat := []
theorem row10753_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10753 row10753_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10753_block_010 : List Nat := []
theorem row10753_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10753 row10753_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10753_block_011 : List Nat := []
theorem row10753_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10753 row10753_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10753_block_012 : List Nat := []
theorem row10753_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10753 row10753_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10753_block_013 : List Nat := []
theorem row10753_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10753 row10753_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10753_large_left_left : List Nat := row10753_small ++ row10753_block_005
theorem row10753_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10753 row10753_large_left_left :=
  Budget.RowUpperBound.branch row10753_small_covers row10753_block_005_covers

def row10753_large_left_right_right : List Nat := row10753_block_007 ++ row10753_block_008
theorem row10753_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10753 row10753_large_left_right_right :=
  Budget.RowUpperBound.branch row10753_block_007_covers row10753_block_008_covers

def row10753_large_left_right : List Nat := row10753_block_006 ++ row10753_large_left_right_right
theorem row10753_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10753 row10753_large_left_right :=
  Budget.RowUpperBound.branch row10753_block_006_covers row10753_large_left_right_right_covers

def row10753_large_left : List Nat := row10753_large_left_left ++ row10753_large_left_right
theorem row10753_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10753 row10753_large_left :=
  Budget.RowUpperBound.branch row10753_large_left_left_covers row10753_large_left_right_covers

def row10753_large_right_left : List Nat := row10753_block_009 ++ row10753_block_010
theorem row10753_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10753 row10753_large_right_left :=
  Budget.RowUpperBound.branch row10753_block_009_covers row10753_block_010_covers

def row10753_large_right_right_right : List Nat := row10753_block_012 ++ row10753_block_013
theorem row10753_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10753 row10753_large_right_right_right :=
  Budget.RowUpperBound.branch row10753_block_012_covers row10753_block_013_covers

def row10753_large_right_right : List Nat := row10753_block_011 ++ row10753_large_right_right_right
theorem row10753_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10753 row10753_large_right_right :=
  Budget.RowUpperBound.branch row10753_block_011_covers row10753_large_right_right_right_covers

def row10753_large_right : List Nat := row10753_large_right_left ++ row10753_large_right_right
theorem row10753_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10753 row10753_large_right :=
  Budget.RowUpperBound.branch row10753_large_right_left_covers row10753_large_right_right_covers

def row10753_large : List Nat := row10753_large_left ++ row10753_large_right
theorem row10753_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10753 row10753_large :=
  Budget.RowUpperBound.branch row10753_large_left_covers row10753_large_right_covers

def row10753 : List Nat := row10753_large
theorem row10753_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10753 row10753 := row10753_large_covers

def row10771_small : List Nat := []
theorem row10771_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10771 row10771_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10771_block_005 : List Nat := []
theorem row10771_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10771 row10771_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10771_block_006 : List Nat := [12409]
def plan10771_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .second, .first, .second, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row10771_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10771 row10771_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10771_block_006
  · decide
  · rfl

def row10771_block_007 : List Nat := [13339, 14011]
def plan10771_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.secondFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .possible, .first, .first, .first, .second, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, .possible, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 31), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10771_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10771 row10771_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10771_block_007
  · decide
  · rfl

def row10771_block_008 : List Nat := [14767]
def plan10771_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .second, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10771_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10771 row10771_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10771_block_008
  · decide
  · rfl

def row10771_block_009 : List Nat := []
theorem row10771_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10771 row10771_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10771_block_010 : List Nat := []
theorem row10771_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10771 row10771_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10771_block_011 : List Nat := []
theorem row10771_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10771 row10771_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10771_block_012 : List Nat := []
theorem row10771_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10771 row10771_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10771_block_013 : List Nat := []
theorem row10771_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10771 row10771_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10771_large_left_left : List Nat := row10771_small ++ row10771_block_005
theorem row10771_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10771 row10771_large_left_left :=
  Budget.RowUpperBound.branch row10771_small_covers row10771_block_005_covers

def row10771_large_left_right_right : List Nat := row10771_block_007 ++ row10771_block_008
theorem row10771_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10771 row10771_large_left_right_right :=
  Budget.RowUpperBound.branch row10771_block_007_covers row10771_block_008_covers

def row10771_large_left_right : List Nat := row10771_block_006 ++ row10771_large_left_right_right
theorem row10771_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10771 row10771_large_left_right :=
  Budget.RowUpperBound.branch row10771_block_006_covers row10771_large_left_right_right_covers

def row10771_large_left : List Nat := row10771_large_left_left ++ row10771_large_left_right
theorem row10771_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10771 row10771_large_left :=
  Budget.RowUpperBound.branch row10771_large_left_left_covers row10771_large_left_right_covers

def row10771_large_right_left : List Nat := row10771_block_009 ++ row10771_block_010
theorem row10771_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10771 row10771_large_right_left :=
  Budget.RowUpperBound.branch row10771_block_009_covers row10771_block_010_covers

def row10771_large_right_right_right : List Nat := row10771_block_012 ++ row10771_block_013
theorem row10771_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10771 row10771_large_right_right_right :=
  Budget.RowUpperBound.branch row10771_block_012_covers row10771_block_013_covers

def row10771_large_right_right : List Nat := row10771_block_011 ++ row10771_large_right_right_right
theorem row10771_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10771 row10771_large_right_right :=
  Budget.RowUpperBound.branch row10771_block_011_covers row10771_large_right_right_right_covers

def row10771_large_right : List Nat := row10771_large_right_left ++ row10771_large_right_right
theorem row10771_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10771 row10771_large_right :=
  Budget.RowUpperBound.branch row10771_large_right_left_covers row10771_large_right_right_covers

def row10771_large : List Nat := row10771_large_left ++ row10771_large_right
theorem row10771_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10771 row10771_large :=
  Budget.RowUpperBound.branch row10771_large_left_covers row10771_large_right_covers

def row10771 : List Nat := row10771_large
theorem row10771_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10771 row10771 := row10771_large_covers

def row10781_small : List Nat := []
theorem row10781_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10781 row10781_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10781_block_005 : List Nat := []
theorem row10781_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10781 row10781_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10781_block_006 : List Nat := [10853, 11369, 11597]
def plan10781_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10781_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10781 row10781_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10781_block_006
  · decide
  · rfl

def row10781_block_007 : List Nat := [12437, 12473, 12491, 12689, 13499, 13721, 14087]
def plan10781_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, .possible, (.secondFactor 13), .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .possible, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 7), .second, .first, (.firstFactor 11), .possible, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10781_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10781 row10781_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10781_block_007
  · decide
  · rfl

def row10781_block_008 : List Nat := []
def plan10781_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 37), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10781_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10781 row10781_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10781_block_008
  · decide
  · rfl

def row10781_block_009 : List Nat := []
theorem row10781_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10781 row10781_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10781_block_010 : List Nat := []
theorem row10781_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10781 row10781_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10781_block_011 : List Nat := []
theorem row10781_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10781 row10781_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10781_block_012 : List Nat := []
theorem row10781_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10781 row10781_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10781_block_013 : List Nat := []
theorem row10781_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10781 row10781_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10781_large_left_left : List Nat := row10781_small ++ row10781_block_005
theorem row10781_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10781 row10781_large_left_left :=
  Budget.RowUpperBound.branch row10781_small_covers row10781_block_005_covers

def row10781_large_left_right_right : List Nat := row10781_block_007 ++ row10781_block_008
theorem row10781_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10781 row10781_large_left_right_right :=
  Budget.RowUpperBound.branch row10781_block_007_covers row10781_block_008_covers

def row10781_large_left_right : List Nat := row10781_block_006 ++ row10781_large_left_right_right
theorem row10781_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10781 row10781_large_left_right :=
  Budget.RowUpperBound.branch row10781_block_006_covers row10781_large_left_right_right_covers

def row10781_large_left : List Nat := row10781_large_left_left ++ row10781_large_left_right
theorem row10781_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10781 row10781_large_left :=
  Budget.RowUpperBound.branch row10781_large_left_left_covers row10781_large_left_right_covers

def row10781_large_right_left : List Nat := row10781_block_009 ++ row10781_block_010
theorem row10781_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10781 row10781_large_right_left :=
  Budget.RowUpperBound.branch row10781_block_009_covers row10781_block_010_covers

def row10781_large_right_right_right : List Nat := row10781_block_012 ++ row10781_block_013
theorem row10781_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10781 row10781_large_right_right_right :=
  Budget.RowUpperBound.branch row10781_block_012_covers row10781_block_013_covers

def row10781_large_right_right : List Nat := row10781_block_011 ++ row10781_large_right_right_right
theorem row10781_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10781 row10781_large_right_right :=
  Budget.RowUpperBound.branch row10781_block_011_covers row10781_large_right_right_right_covers

def row10781_large_right : List Nat := row10781_large_right_left ++ row10781_large_right_right
theorem row10781_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10781 row10781_large_right :=
  Budget.RowUpperBound.branch row10781_large_right_left_covers row10781_large_right_right_covers

def row10781_large : List Nat := row10781_large_left ++ row10781_large_right
theorem row10781_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10781 row10781_large :=
  Budget.RowUpperBound.branch row10781_large_left_covers row10781_large_right_covers

def row10781 : List Nat := row10781_large
theorem row10781_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10781 row10781 := row10781_large_covers

def row10789_small : List Nat := []
theorem row10789_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10789 row10789_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10789_block_005 : List Nat := []
theorem row10789_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10789 row10789_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10789_block_006 : List Nat := [10939]
def plan10789_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row10789_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10789 row10789_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10789_block_006
  · decide
  · rfl

def row10789_block_007 : List Nat := [12487, 13009, 13963, 14461]
def plan10789_block_007 : List Budget.Choice := [.first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .second, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 31), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .possible, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .possible]
set_option maxRecDepth 2048 in
theorem row10789_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10789 row10789_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10789_block_007
  · decide
  · rfl

def row10789_block_008 : List Nat := [14653]
def plan10789_block_008 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 23), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10789_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10789 row10789_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10789_block_008
  · decide
  · rfl

def row10789_block_009 : List Nat := []
theorem row10789_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10789 row10789_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10789_block_010 : List Nat := []
theorem row10789_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10789 row10789_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10789_block_011 : List Nat := []
theorem row10789_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10789 row10789_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10789_block_012 : List Nat := []
theorem row10789_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10789 row10789_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10789_block_013 : List Nat := []
theorem row10789_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10789 row10789_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10789_large_left_left : List Nat := row10789_small ++ row10789_block_005
theorem row10789_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10789 row10789_large_left_left :=
  Budget.RowUpperBound.branch row10789_small_covers row10789_block_005_covers

def row10789_large_left_right_right : List Nat := row10789_block_007 ++ row10789_block_008
theorem row10789_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10789 row10789_large_left_right_right :=
  Budget.RowUpperBound.branch row10789_block_007_covers row10789_block_008_covers

def row10789_large_left_right : List Nat := row10789_block_006 ++ row10789_large_left_right_right
theorem row10789_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10789 row10789_large_left_right :=
  Budget.RowUpperBound.branch row10789_block_006_covers row10789_large_left_right_right_covers

def row10789_large_left : List Nat := row10789_large_left_left ++ row10789_large_left_right
theorem row10789_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10789 row10789_large_left :=
  Budget.RowUpperBound.branch row10789_large_left_left_covers row10789_large_left_right_covers

def row10789_large_right_left : List Nat := row10789_block_009 ++ row10789_block_010
theorem row10789_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10789 row10789_large_right_left :=
  Budget.RowUpperBound.branch row10789_block_009_covers row10789_block_010_covers

def row10789_large_right_right_right : List Nat := row10789_block_012 ++ row10789_block_013
theorem row10789_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10789 row10789_large_right_right_right :=
  Budget.RowUpperBound.branch row10789_block_012_covers row10789_block_013_covers

def row10789_large_right_right : List Nat := row10789_block_011 ++ row10789_large_right_right_right
theorem row10789_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10789 row10789_large_right_right :=
  Budget.RowUpperBound.branch row10789_block_011_covers row10789_large_right_right_right_covers

def row10789_large_right : List Nat := row10789_large_right_left ++ row10789_large_right_right
theorem row10789_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10789 row10789_large_right :=
  Budget.RowUpperBound.branch row10789_large_right_left_covers row10789_large_right_right_covers

def row10789_large : List Nat := row10789_large_left ++ row10789_large_right
theorem row10789_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10789 row10789_large :=
  Budget.RowUpperBound.branch row10789_large_left_covers row10789_large_right_covers

def row10789 : List Nat := row10789_large
theorem row10789_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10789 row10789 := row10789_large_covers

end ProjectEulerSolutions.PrimePairLarge
