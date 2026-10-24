import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8669_small : List Nat := []
theorem row8669_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8669 row8669_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8669_block_005 : List Nat := [8867, 9203, 9551, 9803, 10211]
def plan8669_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 19), .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .possible, (.firstFactor 13), .first, .first, .first, .second, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .possible, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8669_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8669 row8669_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8669_block_005
  · decide
  · rfl

def row8669_block_006 : List Nat := [11489, 11657, 11867, 12401]
def plan8669_block_006 : List Budget.Choice := [(.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), (.secondFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 7), .possible, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 23), .second, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .second]
set_option maxRecDepth 2048 in
theorem row8669_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8669 row8669_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8669_block_006
  · decide
  · rfl

def row8669_block_007 : List Nat := [12983, 13127, 14051]
def plan8669_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .second, .second, .second, (.firstFactor 17), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 23), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8669_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8669 row8669_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8669_block_007
  · decide
  · rfl

def row8669_block_008 : List Nat := [14549, 14627, 16007]
def plan8669_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8669_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8669 row8669_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8669_block_008
  · decide
  · rfl

def row8669_block_009 : List Nat := [17099]
def plan8669_block_009 : List Budget.Choice := [.first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8669_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8669 row8669_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8669_block_009
  · decide
  · rfl

def row8669_block_010 : List Nat := []
theorem row8669_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8669 row8669_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8669_block_011 : List Nat := []
theorem row8669_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8669 row8669_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8669_block_012 : List Nat := []
theorem row8669_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8669 row8669_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8669_block_013 : List Nat := []
theorem row8669_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8669 row8669_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8669_large_left_left : List Nat := row8669_small ++ row8669_block_005
theorem row8669_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8669 row8669_large_left_left :=
  Budget.RowUpperBound.branch row8669_small_covers row8669_block_005_covers

def row8669_large_left_right_right : List Nat := row8669_block_007 ++ row8669_block_008
theorem row8669_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8669 row8669_large_left_right_right :=
  Budget.RowUpperBound.branch row8669_block_007_covers row8669_block_008_covers

def row8669_large_left_right : List Nat := row8669_block_006 ++ row8669_large_left_right_right
theorem row8669_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8669 row8669_large_left_right :=
  Budget.RowUpperBound.branch row8669_block_006_covers row8669_large_left_right_right_covers

def row8669_large_left : List Nat := row8669_large_left_left ++ row8669_large_left_right
theorem row8669_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8669 row8669_large_left :=
  Budget.RowUpperBound.branch row8669_large_left_left_covers row8669_large_left_right_covers

def row8669_large_right_left : List Nat := row8669_block_009 ++ row8669_block_010
theorem row8669_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8669 row8669_large_right_left :=
  Budget.RowUpperBound.branch row8669_block_009_covers row8669_block_010_covers

def row8669_large_right_right_right : List Nat := row8669_block_012 ++ row8669_block_013
theorem row8669_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8669 row8669_large_right_right_right :=
  Budget.RowUpperBound.branch row8669_block_012_covers row8669_block_013_covers

def row8669_large_right_right : List Nat := row8669_block_011 ++ row8669_large_right_right_right
theorem row8669_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8669 row8669_large_right_right :=
  Budget.RowUpperBound.branch row8669_block_011_covers row8669_large_right_right_right_covers

def row8669_large_right : List Nat := row8669_large_right_left ++ row8669_large_right_right
theorem row8669_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8669 row8669_large_right :=
  Budget.RowUpperBound.branch row8669_large_right_left_covers row8669_large_right_right_covers

def row8669_large : List Nat := row8669_large_left ++ row8669_large_right
theorem row8669_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8669 row8669_large :=
  Budget.RowUpperBound.branch row8669_large_left_covers row8669_large_right_covers

def row8669 : List Nat := row8669_large
theorem row8669_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8669 row8669 := row8669_large_covers

def row8677_small : List Nat := []
theorem row8677_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8677 row8677_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8677_block_005 : List Nat := [9103, 9613, 9661]
def plan8677_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), (.secondFactor 31), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 37), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .possible, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 29), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .possible, .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8677_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8677 row8677_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8677_block_005
  · decide
  · rfl

def row8677_block_006 : List Nat := [10399, 10453, 11443, 11701, 11833, 12037, 12253]
def plan8677_block_006 : List Budget.Choice := [.first, .possible, .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 11), (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.firstFactor 37), .first, .possible, (.secondFactor 29), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), .second, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8677_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8677 row8677_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8677_block_006
  · decide
  · rfl

def row8677_block_007 : List Nat := [13009, 13879, 14347]
def plan8677_block_007 : List Budget.Choice := [(.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 31), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 19), .second, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .second, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8677_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8677 row8677_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8677_block_007
  · decide
  · rfl

def row8677_block_008 : List Nat := [15661, 15859]
def plan8677_block_008 : List Budget.Choice := [(.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8677_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8677 row8677_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8677_block_008
  · decide
  · rfl

def row8677_block_009 : List Nat := [17029]
def plan8677_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .second, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .second, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, .first, .second, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8677_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8677 row8677_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8677_block_009
  · decide
  · rfl

def row8677_block_010 : List Nat := []
theorem row8677_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8677 row8677_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8677_block_011 : List Nat := []
theorem row8677_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8677 row8677_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8677_block_012 : List Nat := []
theorem row8677_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8677 row8677_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8677_block_013 : List Nat := []
theorem row8677_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8677 row8677_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8677_large_left_left : List Nat := row8677_small ++ row8677_block_005
theorem row8677_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8677 row8677_large_left_left :=
  Budget.RowUpperBound.branch row8677_small_covers row8677_block_005_covers

def row8677_large_left_right_right : List Nat := row8677_block_007 ++ row8677_block_008
theorem row8677_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8677 row8677_large_left_right_right :=
  Budget.RowUpperBound.branch row8677_block_007_covers row8677_block_008_covers

def row8677_large_left_right : List Nat := row8677_block_006 ++ row8677_large_left_right_right
theorem row8677_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8677 row8677_large_left_right :=
  Budget.RowUpperBound.branch row8677_block_006_covers row8677_large_left_right_right_covers

def row8677_large_left : List Nat := row8677_large_left_left ++ row8677_large_left_right
theorem row8677_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8677 row8677_large_left :=
  Budget.RowUpperBound.branch row8677_large_left_left_covers row8677_large_left_right_covers

def row8677_large_right_left : List Nat := row8677_block_009 ++ row8677_block_010
theorem row8677_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8677 row8677_large_right_left :=
  Budget.RowUpperBound.branch row8677_block_009_covers row8677_block_010_covers

def row8677_large_right_right_right : List Nat := row8677_block_012 ++ row8677_block_013
theorem row8677_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8677 row8677_large_right_right_right :=
  Budget.RowUpperBound.branch row8677_block_012_covers row8677_block_013_covers

def row8677_large_right_right : List Nat := row8677_block_011 ++ row8677_large_right_right_right
theorem row8677_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8677 row8677_large_right_right :=
  Budget.RowUpperBound.branch row8677_block_011_covers row8677_large_right_right_right_covers

def row8677_large_right : List Nat := row8677_large_right_left ++ row8677_large_right_right
theorem row8677_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8677 row8677_large_right :=
  Budget.RowUpperBound.branch row8677_large_right_left_covers row8677_large_right_right_covers

def row8677_large : List Nat := row8677_large_left ++ row8677_large_right
theorem row8677_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8677 row8677_large :=
  Budget.RowUpperBound.branch row8677_large_left_covers row8677_large_right_covers

def row8677 : List Nat := row8677_large
theorem row8677_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8677 row8677 := row8677_large_covers

def row8681_small : List Nat := []
theorem row8681_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8681 row8681_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8681_block_005 : List Nat := [10301]
def plan8681_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .second, .first, .second, .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.secondFactor 29), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8681_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8681 row8681_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8681_block_005
  · decide
  · rfl

def row8681_block_006 : List Nat := [11117, 11243, 11411, 11789]
def plan8681_block_006 : List Budget.Choice := [(.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 29), .possible, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8681_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8681 row8681_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8681_block_006
  · decide
  · rfl

def row8681_block_007 : List Nat := [12911, 13691, 14153]
def plan8681_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 11), .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row8681_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8681 row8681_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8681_block_007
  · decide
  · rfl

def row8681_block_008 : List Nat := [14867, 15161, 15359, 15383, 15581, 16103, 16187]
def plan8681_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 23), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .possible, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .possible, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .second, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 29), .possible, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row8681_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8681 row8681_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8681_block_008
  · decide
  · rfl

def row8681_block_009 : List Nat := [16703]
def plan8681_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 37), .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8681_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8681 row8681_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8681_block_009
  · decide
  · rfl

def row8681_block_010 : List Nat := []
theorem row8681_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8681 row8681_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8681_block_011 : List Nat := []
theorem row8681_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8681 row8681_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8681_block_012 : List Nat := []
theorem row8681_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8681 row8681_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8681_block_013 : List Nat := []
theorem row8681_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8681 row8681_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8681_large_left_left : List Nat := row8681_small ++ row8681_block_005
theorem row8681_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8681 row8681_large_left_left :=
  Budget.RowUpperBound.branch row8681_small_covers row8681_block_005_covers

def row8681_large_left_right_right : List Nat := row8681_block_007 ++ row8681_block_008
theorem row8681_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8681 row8681_large_left_right_right :=
  Budget.RowUpperBound.branch row8681_block_007_covers row8681_block_008_covers

def row8681_large_left_right : List Nat := row8681_block_006 ++ row8681_large_left_right_right
theorem row8681_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8681 row8681_large_left_right :=
  Budget.RowUpperBound.branch row8681_block_006_covers row8681_large_left_right_right_covers

def row8681_large_left : List Nat := row8681_large_left_left ++ row8681_large_left_right
theorem row8681_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8681 row8681_large_left :=
  Budget.RowUpperBound.branch row8681_large_left_left_covers row8681_large_left_right_covers

def row8681_large_right_left : List Nat := row8681_block_009 ++ row8681_block_010
theorem row8681_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8681 row8681_large_right_left :=
  Budget.RowUpperBound.branch row8681_block_009_covers row8681_block_010_covers

def row8681_large_right_right_right : List Nat := row8681_block_012 ++ row8681_block_013
theorem row8681_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8681 row8681_large_right_right_right :=
  Budget.RowUpperBound.branch row8681_block_012_covers row8681_block_013_covers

def row8681_large_right_right : List Nat := row8681_block_011 ++ row8681_large_right_right_right
theorem row8681_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8681 row8681_large_right_right :=
  Budget.RowUpperBound.branch row8681_block_011_covers row8681_large_right_right_right_covers

def row8681_large_right : List Nat := row8681_large_right_left ++ row8681_large_right_right
theorem row8681_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8681 row8681_large_right :=
  Budget.RowUpperBound.branch row8681_large_right_left_covers row8681_large_right_right_covers

def row8681_large : List Nat := row8681_large_left ++ row8681_large_right
theorem row8681_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8681 row8681_large :=
  Budget.RowUpperBound.branch row8681_large_left_covers row8681_large_right_covers

def row8681 : List Nat := row8681_large
theorem row8681_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8681 row8681 := row8681_large_covers

def row8689_small : List Nat := []
theorem row8689_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8689 row8689_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8689_block_005 : List Nat := [8887, 9403, 9781, 10243]
def plan8689_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 23), (.firstFactor 13), .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .second, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 37), .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 13), .first, .first, (.secondFactor 31), (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row8689_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8689 row8689_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8689_block_005
  · decide
  · rfl

def row8689_block_006 : List Nat := [11257, 12301]
def plan8689_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 19), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 19), (.secondFactor 31), (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8689_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8689 row8689_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8689_block_006
  · decide
  · rfl

def row8689_block_007 : List Nat := [13411, 13789, 13879]
def plan8689_block_007 : List Budget.Choice := [(.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .second, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row8689_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8689 row8689_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8689_block_007
  · decide
  · rfl

def row8689_block_008 : List Nat := [14887, 15013]
def plan8689_block_008 : List Budget.Choice := [(.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8689_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8689 row8689_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8689_block_008
  · decide
  · rfl

def row8689_block_009 : List Nat := [16981]
def plan8689_block_009 : List Budget.Choice := [.first, .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 11), .first, .possible, (.firstFactor 19), (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .second, .first, .first, .first, (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8689_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8689 row8689_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8689_block_009
  · decide
  · rfl

def row8689_block_010 : List Nat := []
theorem row8689_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8689 row8689_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8689_block_011 : List Nat := []
theorem row8689_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8689 row8689_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8689_block_012 : List Nat := []
theorem row8689_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8689 row8689_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8689_block_013 : List Nat := []
theorem row8689_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8689 row8689_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8689_large_left_left : List Nat := row8689_small ++ row8689_block_005
theorem row8689_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8689 row8689_large_left_left :=
  Budget.RowUpperBound.branch row8689_small_covers row8689_block_005_covers

def row8689_large_left_right_right : List Nat := row8689_block_007 ++ row8689_block_008
theorem row8689_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8689 row8689_large_left_right_right :=
  Budget.RowUpperBound.branch row8689_block_007_covers row8689_block_008_covers

def row8689_large_left_right : List Nat := row8689_block_006 ++ row8689_large_left_right_right
theorem row8689_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8689 row8689_large_left_right :=
  Budget.RowUpperBound.branch row8689_block_006_covers row8689_large_left_right_right_covers

def row8689_large_left : List Nat := row8689_large_left_left ++ row8689_large_left_right
theorem row8689_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8689 row8689_large_left :=
  Budget.RowUpperBound.branch row8689_large_left_left_covers row8689_large_left_right_covers

def row8689_large_right_left : List Nat := row8689_block_009 ++ row8689_block_010
theorem row8689_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8689 row8689_large_right_left :=
  Budget.RowUpperBound.branch row8689_block_009_covers row8689_block_010_covers

def row8689_large_right_right_right : List Nat := row8689_block_012 ++ row8689_block_013
theorem row8689_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8689 row8689_large_right_right_right :=
  Budget.RowUpperBound.branch row8689_block_012_covers row8689_block_013_covers

def row8689_large_right_right : List Nat := row8689_block_011 ++ row8689_large_right_right_right
theorem row8689_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8689 row8689_large_right_right :=
  Budget.RowUpperBound.branch row8689_block_011_covers row8689_large_right_right_right_covers

def row8689_large_right : List Nat := row8689_large_right_left ++ row8689_large_right_right
theorem row8689_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8689 row8689_large_right :=
  Budget.RowUpperBound.branch row8689_large_right_left_covers row8689_large_right_right_covers

def row8689_large : List Nat := row8689_large_left ++ row8689_large_right
theorem row8689_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8689 row8689_large :=
  Budget.RowUpperBound.branch row8689_large_left_covers row8689_large_right_covers

def row8689 : List Nat := row8689_large
theorem row8689_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8689 row8689 := row8689_large_covers

end ProjectEulerSolutions.PrimePairLarge
