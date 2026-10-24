import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10159_small : List Nat := []
theorem row10159_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10159 row10159_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10159_block_005 : List Nat := []
def plan10159_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row10159_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10159 row10159_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10159_block_005
  · decide
  · rfl

def row10159_block_006 : List Nat := [10723, 11317, 12211]
def plan10159_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .second, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .possible, .second, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .possible, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row10159_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10159 row10159_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10159_block_006
  · decide
  · rfl

def row10159_block_007 : List Nat := [12697, 12841, 13267]
def plan10159_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, .second, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .second, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 23), .possible, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, (.secondFactor 13), (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 13), (.firstFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row10159_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10159 row10159_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10159_block_007
  · decide
  · rfl

def row10159_block_008 : List Nat := [14551, 15217]
def plan10159_block_008 : List Budget.Choice := [(.secondFactor 37), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .possible, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10159_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10159 row10159_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10159_block_008
  · decide
  · rfl

def row10159_block_009 : List Nat := []
theorem row10159_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10159 row10159_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10159_block_010 : List Nat := []
theorem row10159_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10159 row10159_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10159_block_011 : List Nat := []
theorem row10159_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10159 row10159_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10159_block_012 : List Nat := []
theorem row10159_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10159 row10159_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10159_block_013 : List Nat := []
theorem row10159_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10159 row10159_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10159_large_left_left : List Nat := row10159_small ++ row10159_block_005
theorem row10159_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10159 row10159_large_left_left :=
  Budget.RowUpperBound.branch row10159_small_covers row10159_block_005_covers

def row10159_large_left_right_right : List Nat := row10159_block_007 ++ row10159_block_008
theorem row10159_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10159 row10159_large_left_right_right :=
  Budget.RowUpperBound.branch row10159_block_007_covers row10159_block_008_covers

def row10159_large_left_right : List Nat := row10159_block_006 ++ row10159_large_left_right_right
theorem row10159_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10159 row10159_large_left_right :=
  Budget.RowUpperBound.branch row10159_block_006_covers row10159_large_left_right_right_covers

def row10159_large_left : List Nat := row10159_large_left_left ++ row10159_large_left_right
theorem row10159_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10159 row10159_large_left :=
  Budget.RowUpperBound.branch row10159_large_left_left_covers row10159_large_left_right_covers

def row10159_large_right_left : List Nat := row10159_block_009 ++ row10159_block_010
theorem row10159_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10159 row10159_large_right_left :=
  Budget.RowUpperBound.branch row10159_block_009_covers row10159_block_010_covers

def row10159_large_right_right_right : List Nat := row10159_block_012 ++ row10159_block_013
theorem row10159_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10159 row10159_large_right_right_right :=
  Budget.RowUpperBound.branch row10159_block_012_covers row10159_block_013_covers

def row10159_large_right_right : List Nat := row10159_block_011 ++ row10159_large_right_right_right
theorem row10159_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10159 row10159_large_right_right :=
  Budget.RowUpperBound.branch row10159_block_011_covers row10159_large_right_right_right_covers

def row10159_large_right : List Nat := row10159_large_right_left ++ row10159_large_right_right
theorem row10159_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10159 row10159_large_right :=
  Budget.RowUpperBound.branch row10159_large_right_left_covers row10159_large_right_right_covers

def row10159_large : List Nat := row10159_large_left ++ row10159_large_right
theorem row10159_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10159 row10159_large :=
  Budget.RowUpperBound.branch row10159_large_left_covers row10159_large_right_covers

def row10159 : List Nat := row10159_large
theorem row10159_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10159 row10159 := row10159_large_covers

def row10163_small : List Nat := []
theorem row10163_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10163 row10163_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10163_block_005 : List Nat := []
def plan10163_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row10163_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10163 row10163_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10163_block_005
  · decide
  · rfl

def row10163_block_006 : List Nat := [11279, 12203]
def plan10163_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 19), .first, .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10163_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10163 row10163_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10163_block_006
  · decide
  · rfl

def row10163_block_007 : List Nat := [13187]
def plan10163_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row10163_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10163 row10163_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10163_block_007
  · decide
  · rfl

def row10163_block_008 : List Nat := [15551, 15791]
def plan10163_block_008 : List Budget.Choice := [.first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 29), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .possible, .first, .second, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .second, (.secondFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10163_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10163 row10163_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10163_block_008
  · decide
  · rfl

def row10163_block_009 : List Nat := []
theorem row10163_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10163 row10163_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10163_block_010 : List Nat := []
theorem row10163_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10163 row10163_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10163_block_011 : List Nat := []
theorem row10163_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10163 row10163_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10163_block_012 : List Nat := []
theorem row10163_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10163 row10163_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10163_block_013 : List Nat := []
theorem row10163_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10163 row10163_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10163_large_left_left : List Nat := row10163_small ++ row10163_block_005
theorem row10163_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10163 row10163_large_left_left :=
  Budget.RowUpperBound.branch row10163_small_covers row10163_block_005_covers

def row10163_large_left_right_right : List Nat := row10163_block_007 ++ row10163_block_008
theorem row10163_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10163 row10163_large_left_right_right :=
  Budget.RowUpperBound.branch row10163_block_007_covers row10163_block_008_covers

def row10163_large_left_right : List Nat := row10163_block_006 ++ row10163_large_left_right_right
theorem row10163_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10163 row10163_large_left_right :=
  Budget.RowUpperBound.branch row10163_block_006_covers row10163_large_left_right_right_covers

def row10163_large_left : List Nat := row10163_large_left_left ++ row10163_large_left_right
theorem row10163_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10163 row10163_large_left :=
  Budget.RowUpperBound.branch row10163_large_left_left_covers row10163_large_left_right_covers

def row10163_large_right_left : List Nat := row10163_block_009 ++ row10163_block_010
theorem row10163_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10163 row10163_large_right_left :=
  Budget.RowUpperBound.branch row10163_block_009_covers row10163_block_010_covers

def row10163_large_right_right_right : List Nat := row10163_block_012 ++ row10163_block_013
theorem row10163_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10163 row10163_large_right_right_right :=
  Budget.RowUpperBound.branch row10163_block_012_covers row10163_block_013_covers

def row10163_large_right_right : List Nat := row10163_block_011 ++ row10163_large_right_right_right
theorem row10163_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10163 row10163_large_right_right :=
  Budget.RowUpperBound.branch row10163_block_011_covers row10163_large_right_right_right_covers

def row10163_large_right : List Nat := row10163_large_right_left ++ row10163_large_right_right
theorem row10163_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10163 row10163_large_right :=
  Budget.RowUpperBound.branch row10163_large_right_left_covers row10163_large_right_right_covers

def row10163_large : List Nat := row10163_large_left ++ row10163_large_right
theorem row10163_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10163 row10163_large :=
  Budget.RowUpperBound.branch row10163_large_left_covers row10163_large_right_covers

def row10163 : List Nat := row10163_large
theorem row10163_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10163 row10163 := row10163_large_covers

def row10169_small : List Nat := []
theorem row10169_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10169 row10169_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10169_block_005 : List Nat := []
def plan10169_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row10169_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10169 row10169_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10169_block_005
  · decide
  · rfl

def row10169_block_006 : List Nat := [11261, 11393, 11447]
def plan10169_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 29), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row10169_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10169 row10169_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10169_block_006
  · decide
  · rfl

def row10169_block_007 : List Nat := [13619, 13967, 14207]
def plan10169_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .second, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .second, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 37), .first, .first, (.secondFactor 17), .second, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row10169_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10169 row10169_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10169_block_007
  · decide
  · rfl

def row10169_block_008 : List Nat := [14537]
def plan10169_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 37), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 23), .first, .second, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .second, .first, (.secondFactor 17), .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10169_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10169 row10169_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10169_block_008
  · decide
  · rfl

def row10169_block_009 : List Nat := []
theorem row10169_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10169 row10169_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10169_block_010 : List Nat := []
theorem row10169_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10169 row10169_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10169_block_011 : List Nat := []
theorem row10169_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10169 row10169_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10169_block_012 : List Nat := []
theorem row10169_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10169 row10169_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10169_block_013 : List Nat := []
theorem row10169_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10169 row10169_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10169_large_left_left : List Nat := row10169_small ++ row10169_block_005
theorem row10169_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10169 row10169_large_left_left :=
  Budget.RowUpperBound.branch row10169_small_covers row10169_block_005_covers

def row10169_large_left_right_right : List Nat := row10169_block_007 ++ row10169_block_008
theorem row10169_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10169 row10169_large_left_right_right :=
  Budget.RowUpperBound.branch row10169_block_007_covers row10169_block_008_covers

def row10169_large_left_right : List Nat := row10169_block_006 ++ row10169_large_left_right_right
theorem row10169_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10169 row10169_large_left_right :=
  Budget.RowUpperBound.branch row10169_block_006_covers row10169_large_left_right_right_covers

def row10169_large_left : List Nat := row10169_large_left_left ++ row10169_large_left_right
theorem row10169_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10169 row10169_large_left :=
  Budget.RowUpperBound.branch row10169_large_left_left_covers row10169_large_left_right_covers

def row10169_large_right_left : List Nat := row10169_block_009 ++ row10169_block_010
theorem row10169_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10169 row10169_large_right_left :=
  Budget.RowUpperBound.branch row10169_block_009_covers row10169_block_010_covers

def row10169_large_right_right_right : List Nat := row10169_block_012 ++ row10169_block_013
theorem row10169_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10169 row10169_large_right_right_right :=
  Budget.RowUpperBound.branch row10169_block_012_covers row10169_block_013_covers

def row10169_large_right_right : List Nat := row10169_block_011 ++ row10169_large_right_right_right
theorem row10169_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10169 row10169_large_right_right :=
  Budget.RowUpperBound.branch row10169_block_011_covers row10169_large_right_right_right_covers

def row10169_large_right : List Nat := row10169_large_right_left ++ row10169_large_right_right
theorem row10169_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10169 row10169_large_right :=
  Budget.RowUpperBound.branch row10169_large_right_left_covers row10169_large_right_right_covers

def row10169_large : List Nat := row10169_large_left ++ row10169_large_right
theorem row10169_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10169 row10169_large :=
  Budget.RowUpperBound.branch row10169_large_left_covers row10169_large_right_covers

def row10169 : List Nat := row10169_large
theorem row10169_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10169 row10169 := row10169_large_covers

def row10177_small : List Nat := []
theorem row10177_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10177 row10177_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10177_block_005 : List Nat := []
def plan10177_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10177_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10177 row10177_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10177_block_005
  · decide
  · rfl

def row10177_block_006 : List Nat := [11131, 11923]
def plan10177_block_006 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 29), (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .second, .first, .first, .first, (.secondFactor 19), (.secondFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10177_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10177 row10177_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10177_block_006
  · decide
  · rfl

def row10177_block_007 : List Nat := [13063, 13441, 13693, 14083]
def plan10177_block_007 : List Budget.Choice := [(.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 19), .first, .first, (.secondFactor 31), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, .second, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 17), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .second, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row10177_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10177 row10177_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10177_block_007
  · decide
  · rfl

def row10177_block_008 : List Nat := [14767]
def plan10177_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 37), (.firstFactor 11), (.secondFactor 17), (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 17), .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10177_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10177 row10177_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10177_block_008
  · decide
  · rfl

def row10177_block_009 : List Nat := []
theorem row10177_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10177 row10177_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10177_block_010 : List Nat := []
theorem row10177_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10177 row10177_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10177_block_011 : List Nat := []
theorem row10177_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10177 row10177_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10177_block_012 : List Nat := []
theorem row10177_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10177 row10177_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10177_block_013 : List Nat := []
theorem row10177_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10177 row10177_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10177_large_left_left : List Nat := row10177_small ++ row10177_block_005
theorem row10177_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10177 row10177_large_left_left :=
  Budget.RowUpperBound.branch row10177_small_covers row10177_block_005_covers

def row10177_large_left_right_right : List Nat := row10177_block_007 ++ row10177_block_008
theorem row10177_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10177 row10177_large_left_right_right :=
  Budget.RowUpperBound.branch row10177_block_007_covers row10177_block_008_covers

def row10177_large_left_right : List Nat := row10177_block_006 ++ row10177_large_left_right_right
theorem row10177_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10177 row10177_large_left_right :=
  Budget.RowUpperBound.branch row10177_block_006_covers row10177_large_left_right_right_covers

def row10177_large_left : List Nat := row10177_large_left_left ++ row10177_large_left_right
theorem row10177_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10177 row10177_large_left :=
  Budget.RowUpperBound.branch row10177_large_left_left_covers row10177_large_left_right_covers

def row10177_large_right_left : List Nat := row10177_block_009 ++ row10177_block_010
theorem row10177_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10177 row10177_large_right_left :=
  Budget.RowUpperBound.branch row10177_block_009_covers row10177_block_010_covers

def row10177_large_right_right_right : List Nat := row10177_block_012 ++ row10177_block_013
theorem row10177_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10177 row10177_large_right_right_right :=
  Budget.RowUpperBound.branch row10177_block_012_covers row10177_block_013_covers

def row10177_large_right_right : List Nat := row10177_block_011 ++ row10177_large_right_right_right
theorem row10177_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10177 row10177_large_right_right :=
  Budget.RowUpperBound.branch row10177_block_011_covers row10177_large_right_right_right_covers

def row10177_large_right : List Nat := row10177_large_right_left ++ row10177_large_right_right
theorem row10177_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10177 row10177_large_right :=
  Budget.RowUpperBound.branch row10177_large_right_left_covers row10177_large_right_right_covers

def row10177_large : List Nat := row10177_large_left ++ row10177_large_right
theorem row10177_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10177 row10177_large :=
  Budget.RowUpperBound.branch row10177_large_left_covers row10177_large_right_covers

def row10177 : List Nat := row10177_large
theorem row10177_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10177 row10177 := row10177_large_covers

end ProjectEulerSolutions.PrimePairLarge
