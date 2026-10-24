import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8527_small : List Nat := []
theorem row8527_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8527 row8527_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8527_block_005 : List Nat := [8929, 9433, 9613, 9907]
def plan8527_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 13), .second, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 31), .possible, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 19), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .second, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .second, .first, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8527_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8527 row8527_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8527_block_005
  · decide
  · rfl

def row8527_block_006 : List Nat := [12253]
def plan8527_block_006 : List Budget.Choice := [.first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .second, (.firstFactor 11), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 17), (.firstFactor 37), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8527_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8527 row8527_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8527_block_006
  · decide
  · rfl

def row8527_block_007 : List Nat := [13093]
def plan8527_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .second, (.secondFactor 23), (.firstFactor 13), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8527_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8527 row8527_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8527_block_007
  · decide
  · rfl

def row8527_block_008 : List Nat := [14557]
def plan8527_block_008 : List Budget.Choice := [.first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .possible, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 31), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8527_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8527 row8527_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8527_block_008
  · decide
  · rfl

def row8527_block_009 : List Nat := [16741, 17209]
def plan8527_block_009 : List Budget.Choice := [(.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 23), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8527_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8527 row8527_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8527_block_009
  · decide
  · rfl

def row8527_block_010 : List Nat := []
theorem row8527_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8527 row8527_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8527_block_011 : List Nat := []
theorem row8527_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8527 row8527_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8527_block_012 : List Nat := []
theorem row8527_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8527 row8527_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8527_block_013 : List Nat := []
theorem row8527_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8527 row8527_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8527_large_left_left : List Nat := row8527_small ++ row8527_block_005
theorem row8527_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8527 row8527_large_left_left :=
  Budget.RowUpperBound.branch row8527_small_covers row8527_block_005_covers

def row8527_large_left_right_right : List Nat := row8527_block_007 ++ row8527_block_008
theorem row8527_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8527 row8527_large_left_right_right :=
  Budget.RowUpperBound.branch row8527_block_007_covers row8527_block_008_covers

def row8527_large_left_right : List Nat := row8527_block_006 ++ row8527_large_left_right_right
theorem row8527_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8527 row8527_large_left_right :=
  Budget.RowUpperBound.branch row8527_block_006_covers row8527_large_left_right_right_covers

def row8527_large_left : List Nat := row8527_large_left_left ++ row8527_large_left_right
theorem row8527_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8527 row8527_large_left :=
  Budget.RowUpperBound.branch row8527_large_left_left_covers row8527_large_left_right_covers

def row8527_large_right_left : List Nat := row8527_block_009 ++ row8527_block_010
theorem row8527_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8527 row8527_large_right_left :=
  Budget.RowUpperBound.branch row8527_block_009_covers row8527_block_010_covers

def row8527_large_right_right_right : List Nat := row8527_block_012 ++ row8527_block_013
theorem row8527_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8527 row8527_large_right_right_right :=
  Budget.RowUpperBound.branch row8527_block_012_covers row8527_block_013_covers

def row8527_large_right_right : List Nat := row8527_block_011 ++ row8527_large_right_right_right
theorem row8527_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8527 row8527_large_right_right :=
  Budget.RowUpperBound.branch row8527_block_011_covers row8527_large_right_right_right_covers

def row8527_large_right : List Nat := row8527_large_right_left ++ row8527_large_right_right
theorem row8527_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8527 row8527_large_right :=
  Budget.RowUpperBound.branch row8527_large_right_left_covers row8527_large_right_right_covers

def row8527_large : List Nat := row8527_large_left ++ row8527_large_right
theorem row8527_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8527 row8527_large :=
  Budget.RowUpperBound.branch row8527_large_left_covers row8527_large_right_covers

def row8527 : List Nat := row8527_large
theorem row8527_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8527 row8527 := row8527_large_covers

def row8537_small : List Nat := []
theorem row8537_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8537 row8537_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8537_block_005 : List Nat := [8753, 8963, 8999, 9377, 9551, 9629]
def plan8537_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 31), .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .possible, .second, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row8537_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8537 row8537_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8537_block_005
  · decide
  · rfl

def row8537_block_006 : List Nat := [10391, 11093, 11279, 11519, 11867, 12323]
def plan8537_block_006 : List Budget.Choice := [.possible, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), .possible, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row8537_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8537 row8537_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8537_block_006
  · decide
  · rfl

def row8537_block_007 : List Nat := [14009]
def plan8537_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .second, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8537_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8537 row8537_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8537_block_007
  · decide
  · rfl

def row8537_block_008 : List Nat := [14639, 14747, 15017, 15761, 15881]
def plan8537_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 11), .possible, (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 31), .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8537_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8537 row8537_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8537_block_008
  · decide
  · rfl

def row8537_block_009 : List Nat := [16979, 17351]
def plan8537_block_009 : List Budget.Choice := [.first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .second, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 19), (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .second, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8537_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8537 row8537_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8537_block_009
  · decide
  · rfl

def row8537_block_010 : List Nat := []
theorem row8537_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8537 row8537_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8537_block_011 : List Nat := []
theorem row8537_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8537 row8537_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8537_block_012 : List Nat := []
theorem row8537_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8537 row8537_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8537_block_013 : List Nat := []
theorem row8537_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8537 row8537_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8537_large_left_left : List Nat := row8537_small ++ row8537_block_005
theorem row8537_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8537 row8537_large_left_left :=
  Budget.RowUpperBound.branch row8537_small_covers row8537_block_005_covers

def row8537_large_left_right_right : List Nat := row8537_block_007 ++ row8537_block_008
theorem row8537_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8537 row8537_large_left_right_right :=
  Budget.RowUpperBound.branch row8537_block_007_covers row8537_block_008_covers

def row8537_large_left_right : List Nat := row8537_block_006 ++ row8537_large_left_right_right
theorem row8537_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8537 row8537_large_left_right :=
  Budget.RowUpperBound.branch row8537_block_006_covers row8537_large_left_right_right_covers

def row8537_large_left : List Nat := row8537_large_left_left ++ row8537_large_left_right
theorem row8537_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8537 row8537_large_left :=
  Budget.RowUpperBound.branch row8537_large_left_left_covers row8537_large_left_right_covers

def row8537_large_right_left : List Nat := row8537_block_009 ++ row8537_block_010
theorem row8537_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8537 row8537_large_right_left :=
  Budget.RowUpperBound.branch row8537_block_009_covers row8537_block_010_covers

def row8537_large_right_right_right : List Nat := row8537_block_012 ++ row8537_block_013
theorem row8537_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8537 row8537_large_right_right_right :=
  Budget.RowUpperBound.branch row8537_block_012_covers row8537_block_013_covers

def row8537_large_right_right : List Nat := row8537_block_011 ++ row8537_large_right_right_right
theorem row8537_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8537 row8537_large_right_right :=
  Budget.RowUpperBound.branch row8537_block_011_covers row8537_large_right_right_right_covers

def row8537_large_right : List Nat := row8537_large_right_left ++ row8537_large_right_right
theorem row8537_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8537 row8537_large_right :=
  Budget.RowUpperBound.branch row8537_large_right_left_covers row8537_large_right_right_covers

def row8537_large : List Nat := row8537_large_left ++ row8537_large_right
theorem row8537_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8537 row8537_large :=
  Budget.RowUpperBound.branch row8537_large_left_covers row8537_large_right_covers

def row8537 : List Nat := row8537_large
theorem row8537_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8537 row8537 := row8537_large_covers

def row8539_small : List Nat := []
theorem row8539_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8539 row8539_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8539_block_005 : List Nat := [9343, 9901]
def plan8539_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 31), .first, .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 29), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8539_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8539 row8539_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8539_block_005
  · decide
  · rfl

def row8539_block_006 : List Nat := []
def plan8539_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .second, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8539_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8539 row8539_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8539_block_006
  · decide
  · rfl

def row8539_block_007 : List Nat := [12853]
def plan8539_block_007 : List Budget.Choice := [(.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .second, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8539_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8539 row8539_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8539_block_007
  · decide
  · rfl

def row8539_block_008 : List Nat := [14563, 15061, 15607]
def plan8539_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .second, (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 17), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 31), (.firstFactor 37), .second, .first, .first, (.secondFactor 19), .second, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8539_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8539 row8539_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8539_block_008
  · decide
  · rfl

def row8539_block_009 : List Nat := [17107, 17359]
def plan8539_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .second, (.firstFactor 37), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, .first, .first, .second, .first, .second, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8539_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8539 row8539_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8539_block_009
  · decide
  · rfl

def row8539_block_010 : List Nat := []
theorem row8539_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8539 row8539_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8539_block_011 : List Nat := []
theorem row8539_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8539 row8539_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8539_block_012 : List Nat := []
theorem row8539_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8539 row8539_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8539_block_013 : List Nat := []
theorem row8539_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8539 row8539_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8539_large_left_left : List Nat := row8539_small ++ row8539_block_005
theorem row8539_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8539 row8539_large_left_left :=
  Budget.RowUpperBound.branch row8539_small_covers row8539_block_005_covers

def row8539_large_left_right_right : List Nat := row8539_block_007 ++ row8539_block_008
theorem row8539_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8539 row8539_large_left_right_right :=
  Budget.RowUpperBound.branch row8539_block_007_covers row8539_block_008_covers

def row8539_large_left_right : List Nat := row8539_block_006 ++ row8539_large_left_right_right
theorem row8539_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8539 row8539_large_left_right :=
  Budget.RowUpperBound.branch row8539_block_006_covers row8539_large_left_right_right_covers

def row8539_large_left : List Nat := row8539_large_left_left ++ row8539_large_left_right
theorem row8539_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8539 row8539_large_left :=
  Budget.RowUpperBound.branch row8539_large_left_left_covers row8539_large_left_right_covers

def row8539_large_right_left : List Nat := row8539_block_009 ++ row8539_block_010
theorem row8539_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8539 row8539_large_right_left :=
  Budget.RowUpperBound.branch row8539_block_009_covers row8539_block_010_covers

def row8539_large_right_right_right : List Nat := row8539_block_012 ++ row8539_block_013
theorem row8539_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8539 row8539_large_right_right_right :=
  Budget.RowUpperBound.branch row8539_block_012_covers row8539_block_013_covers

def row8539_large_right_right : List Nat := row8539_block_011 ++ row8539_large_right_right_right
theorem row8539_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8539 row8539_large_right_right :=
  Budget.RowUpperBound.branch row8539_block_011_covers row8539_large_right_right_right_covers

def row8539_large_right : List Nat := row8539_large_right_left ++ row8539_large_right_right
theorem row8539_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8539 row8539_large_right :=
  Budget.RowUpperBound.branch row8539_large_right_left_covers row8539_large_right_right_covers

def row8539_large : List Nat := row8539_large_left ++ row8539_large_right
theorem row8539_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8539 row8539_large :=
  Budget.RowUpperBound.branch row8539_large_left_covers row8539_large_right_covers

def row8539 : List Nat := row8539_large
theorem row8539_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8539 row8539 := row8539_large_covers

def row8543_small : List Nat := []
theorem row8543_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8543 row8543_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8543_block_005 : List Nat := [8609, 9281, 9791, 9803, 10169]
def plan8543_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 31), .second, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 23), .second, .first, (.firstFactor 19), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .second, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .possible, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8543_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8543 row8543_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8543_block_005
  · decide
  · rfl

def row8543_block_006 : List Nat := [10463, 10859, 11351, 11519, 12119, 12323, 12413]
def plan8543_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 31), .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .possible, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 17), (.secondFactor 11), .first, .first, .first, .first, .second, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .possible, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .possible]
set_option maxRecDepth 2048 in
theorem row8543_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8543 row8543_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8543_block_006
  · decide
  · rfl

def row8543_block_007 : List Nat := [13337, 13751, 13907, 14207]
def plan8543_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 29), (.secondFactor 29), (.secondFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8543_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8543 row8543_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8543_block_007
  · decide
  · rfl

def row8543_block_008 : List Nat := [14969]
def plan8543_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 17), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8543_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8543 row8543_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8543_block_008
  · decide
  · rfl

def row8543_block_009 : List Nat := [16937]
def plan8543_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .second, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8543_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8543 row8543_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8543_block_009
  · decide
  · rfl

def row8543_block_010 : List Nat := []
theorem row8543_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8543 row8543_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8543_block_011 : List Nat := []
theorem row8543_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8543 row8543_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8543_block_012 : List Nat := []
theorem row8543_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8543 row8543_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8543_block_013 : List Nat := []
theorem row8543_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8543 row8543_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8543_large_left_left : List Nat := row8543_small ++ row8543_block_005
theorem row8543_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8543 row8543_large_left_left :=
  Budget.RowUpperBound.branch row8543_small_covers row8543_block_005_covers

def row8543_large_left_right_right : List Nat := row8543_block_007 ++ row8543_block_008
theorem row8543_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8543 row8543_large_left_right_right :=
  Budget.RowUpperBound.branch row8543_block_007_covers row8543_block_008_covers

def row8543_large_left_right : List Nat := row8543_block_006 ++ row8543_large_left_right_right
theorem row8543_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8543 row8543_large_left_right :=
  Budget.RowUpperBound.branch row8543_block_006_covers row8543_large_left_right_right_covers

def row8543_large_left : List Nat := row8543_large_left_left ++ row8543_large_left_right
theorem row8543_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8543 row8543_large_left :=
  Budget.RowUpperBound.branch row8543_large_left_left_covers row8543_large_left_right_covers

def row8543_large_right_left : List Nat := row8543_block_009 ++ row8543_block_010
theorem row8543_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8543 row8543_large_right_left :=
  Budget.RowUpperBound.branch row8543_block_009_covers row8543_block_010_covers

def row8543_large_right_right_right : List Nat := row8543_block_012 ++ row8543_block_013
theorem row8543_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8543 row8543_large_right_right_right :=
  Budget.RowUpperBound.branch row8543_block_012_covers row8543_block_013_covers

def row8543_large_right_right : List Nat := row8543_block_011 ++ row8543_large_right_right_right
theorem row8543_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8543 row8543_large_right_right :=
  Budget.RowUpperBound.branch row8543_block_011_covers row8543_large_right_right_right_covers

def row8543_large_right : List Nat := row8543_large_right_left ++ row8543_large_right_right
theorem row8543_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8543 row8543_large_right :=
  Budget.RowUpperBound.branch row8543_large_right_left_covers row8543_large_right_right_covers

def row8543_large : List Nat := row8543_large_left ++ row8543_large_right
theorem row8543_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8543 row8543_large :=
  Budget.RowUpperBound.branch row8543_large_left_covers row8543_large_right_covers

def row8543 : List Nat := row8543_large
theorem row8543_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8543 row8543 := row8543_large_covers

end ProjectEulerSolutions.PrimePairLarge
