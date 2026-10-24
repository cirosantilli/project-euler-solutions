import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8933_small : List Nat := []
theorem row8933_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8933 row8933_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8933_block_005 : List Nat := [9377, 9461, 9491]
def plan8933_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .second, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 19), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 19), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8933_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8933 row8933_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8933_block_005
  · decide
  · rfl

def row8933_block_006 : List Nat := [10739, 11213, 12071]
def plan8933_block_006 : List Budget.Choice := [.second, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .second, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 17), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .second, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8933_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8933 row8933_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8933_block_006
  · decide
  · rfl

def row8933_block_007 : List Nat := [13103, 13451]
def plan8933_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .possible, (.firstFactor 13), (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row8933_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8933 row8933_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8933_block_007
  · decide
  · rfl

def row8933_block_008 : List Nat := [14951, 15233, 15581, 15767, 16421]
def plan8933_block_008 : List Budget.Choice := [.first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.secondFactor 13), .possible, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .second, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8933_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8933 row8933_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8933_block_008
  · decide
  · rfl

def row8933_block_009 : List Nat := []
def plan8933_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .second, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), (.secondFactor 11), .first, .first, .second, .first, .second, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8933_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8933 row8933_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8933_block_009
  · decide
  · rfl

def row8933_block_010 : List Nat := []
theorem row8933_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8933 row8933_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8933_block_011 : List Nat := []
theorem row8933_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8933 row8933_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8933_block_012 : List Nat := []
theorem row8933_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8933 row8933_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8933_block_013 : List Nat := []
theorem row8933_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8933 row8933_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8933_large_left_left : List Nat := row8933_small ++ row8933_block_005
theorem row8933_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8933 row8933_large_left_left :=
  Budget.RowUpperBound.branch row8933_small_covers row8933_block_005_covers

def row8933_large_left_right_right : List Nat := row8933_block_007 ++ row8933_block_008
theorem row8933_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8933 row8933_large_left_right_right :=
  Budget.RowUpperBound.branch row8933_block_007_covers row8933_block_008_covers

def row8933_large_left_right : List Nat := row8933_block_006 ++ row8933_large_left_right_right
theorem row8933_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8933 row8933_large_left_right :=
  Budget.RowUpperBound.branch row8933_block_006_covers row8933_large_left_right_right_covers

def row8933_large_left : List Nat := row8933_large_left_left ++ row8933_large_left_right
theorem row8933_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8933 row8933_large_left :=
  Budget.RowUpperBound.branch row8933_large_left_left_covers row8933_large_left_right_covers

def row8933_large_right_left : List Nat := row8933_block_009 ++ row8933_block_010
theorem row8933_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8933 row8933_large_right_left :=
  Budget.RowUpperBound.branch row8933_block_009_covers row8933_block_010_covers

def row8933_large_right_right_right : List Nat := row8933_block_012 ++ row8933_block_013
theorem row8933_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8933 row8933_large_right_right_right :=
  Budget.RowUpperBound.branch row8933_block_012_covers row8933_block_013_covers

def row8933_large_right_right : List Nat := row8933_block_011 ++ row8933_large_right_right_right
theorem row8933_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8933 row8933_large_right_right :=
  Budget.RowUpperBound.branch row8933_block_011_covers row8933_large_right_right_right_covers

def row8933_large_right : List Nat := row8933_large_right_left ++ row8933_large_right_right
theorem row8933_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8933 row8933_large_right :=
  Budget.RowUpperBound.branch row8933_large_right_left_covers row8933_large_right_right_covers

def row8933_large : List Nat := row8933_large_left ++ row8933_large_right
theorem row8933_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8933 row8933_large :=
  Budget.RowUpperBound.branch row8933_large_left_covers row8933_large_right_covers

def row8933 : List Nat := row8933_large
theorem row8933_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8933 row8933 := row8933_large_covers

def row8941_small : List Nat := []
theorem row8941_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8941 row8941_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8941_block_005 : List Nat := [9067]
def plan8941_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .possible, (.secondFactor 17), (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 29), .second, .first, .first, (.secondFactor 37), (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8941_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8941 row8941_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8941_block_005
  · decide
  · rfl

def row8941_block_006 : List Nat := [10831]
def plan8941_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .possible, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8941_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8941 row8941_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8941_block_006
  · decide
  · rfl

def row8941_block_007 : List Nat := [14449]
def plan8941_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .second]
set_option maxRecDepth 2048 in
theorem row8941_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8941 row8941_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8941_block_007
  · decide
  · rfl

def row8941_block_008 : List Nat := [14629, 14869, 15307, 15973, 16417]
def plan8941_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .second, .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 7), .second, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 7), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8941_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8941 row8941_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8941_block_008
  · decide
  · rfl

def row8941_block_009 : List Nat := [16567]
def plan8941_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8941_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8941 row8941_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8941_block_009
  · decide
  · rfl

def row8941_block_010 : List Nat := []
theorem row8941_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8941 row8941_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8941_block_011 : List Nat := []
theorem row8941_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8941 row8941_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8941_block_012 : List Nat := []
theorem row8941_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8941 row8941_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8941_block_013 : List Nat := []
theorem row8941_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8941 row8941_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8941_large_left_left : List Nat := row8941_small ++ row8941_block_005
theorem row8941_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8941 row8941_large_left_left :=
  Budget.RowUpperBound.branch row8941_small_covers row8941_block_005_covers

def row8941_large_left_right_right : List Nat := row8941_block_007 ++ row8941_block_008
theorem row8941_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8941 row8941_large_left_right_right :=
  Budget.RowUpperBound.branch row8941_block_007_covers row8941_block_008_covers

def row8941_large_left_right : List Nat := row8941_block_006 ++ row8941_large_left_right_right
theorem row8941_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8941 row8941_large_left_right :=
  Budget.RowUpperBound.branch row8941_block_006_covers row8941_large_left_right_right_covers

def row8941_large_left : List Nat := row8941_large_left_left ++ row8941_large_left_right
theorem row8941_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8941 row8941_large_left :=
  Budget.RowUpperBound.branch row8941_large_left_left_covers row8941_large_left_right_covers

def row8941_large_right_left : List Nat := row8941_block_009 ++ row8941_block_010
theorem row8941_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8941 row8941_large_right_left :=
  Budget.RowUpperBound.branch row8941_block_009_covers row8941_block_010_covers

def row8941_large_right_right_right : List Nat := row8941_block_012 ++ row8941_block_013
theorem row8941_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8941 row8941_large_right_right_right :=
  Budget.RowUpperBound.branch row8941_block_012_covers row8941_block_013_covers

def row8941_large_right_right : List Nat := row8941_block_011 ++ row8941_large_right_right_right
theorem row8941_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8941 row8941_large_right_right :=
  Budget.RowUpperBound.branch row8941_block_011_covers row8941_large_right_right_right_covers

def row8941_large_right : List Nat := row8941_large_right_left ++ row8941_large_right_right
theorem row8941_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8941 row8941_large_right :=
  Budget.RowUpperBound.branch row8941_large_right_left_covers row8941_large_right_right_covers

def row8941_large : List Nat := row8941_large_left ++ row8941_large_right
theorem row8941_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8941 row8941_large :=
  Budget.RowUpperBound.branch row8941_large_left_covers row8941_large_right_covers

def row8941 : List Nat := row8941_large
theorem row8941_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8941 row8941 := row8941_large_covers

def row8951_small : List Nat := []
theorem row8951_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8951 row8951_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8951_block_005 : List Nat := [10193]
def plan8951_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .possible, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 37), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row8951_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8951 row8951_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8951_block_005
  · decide
  · rfl

def row8951_block_006 : List Nat := [10499, 10631, 10949, 11423, 12143]
def plan8951_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .second, .possible, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 23), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 29), (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8951_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8951 row8951_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8951_block_006
  · decide
  · rfl

def row8951_block_007 : List Nat := [12809, 13001, 13679, 13859, 14177, 14321, 14327]
def plan8951_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 17), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .possible, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .possible, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .possible, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8951_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8951 row8951_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8951_block_007
  · decide
  · rfl

def row8951_block_008 : List Nat := [15101, 16301]
def plan8951_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .possible, (.secondFactor 29), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row8951_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8951 row8951_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8951_block_008
  · decide
  · rfl

def row8951_block_009 : List Nat := []
def plan8951_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8951_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8951 row8951_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8951_block_009
  · decide
  · rfl

def row8951_block_010 : List Nat := []
theorem row8951_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8951 row8951_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8951_block_011 : List Nat := []
theorem row8951_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8951 row8951_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8951_block_012 : List Nat := []
theorem row8951_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8951 row8951_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8951_block_013 : List Nat := []
theorem row8951_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8951 row8951_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8951_large_left_left : List Nat := row8951_small ++ row8951_block_005
theorem row8951_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8951 row8951_large_left_left :=
  Budget.RowUpperBound.branch row8951_small_covers row8951_block_005_covers

def row8951_large_left_right_right : List Nat := row8951_block_007 ++ row8951_block_008
theorem row8951_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8951 row8951_large_left_right_right :=
  Budget.RowUpperBound.branch row8951_block_007_covers row8951_block_008_covers

def row8951_large_left_right : List Nat := row8951_block_006 ++ row8951_large_left_right_right
theorem row8951_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8951 row8951_large_left_right :=
  Budget.RowUpperBound.branch row8951_block_006_covers row8951_large_left_right_right_covers

def row8951_large_left : List Nat := row8951_large_left_left ++ row8951_large_left_right
theorem row8951_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8951 row8951_large_left :=
  Budget.RowUpperBound.branch row8951_large_left_left_covers row8951_large_left_right_covers

def row8951_large_right_left : List Nat := row8951_block_009 ++ row8951_block_010
theorem row8951_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8951 row8951_large_right_left :=
  Budget.RowUpperBound.branch row8951_block_009_covers row8951_block_010_covers

def row8951_large_right_right_right : List Nat := row8951_block_012 ++ row8951_block_013
theorem row8951_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8951 row8951_large_right_right_right :=
  Budget.RowUpperBound.branch row8951_block_012_covers row8951_block_013_covers

def row8951_large_right_right : List Nat := row8951_block_011 ++ row8951_large_right_right_right
theorem row8951_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8951 row8951_large_right_right :=
  Budget.RowUpperBound.branch row8951_block_011_covers row8951_large_right_right_right_covers

def row8951_large_right : List Nat := row8951_large_right_left ++ row8951_large_right_right
theorem row8951_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8951 row8951_large_right :=
  Budget.RowUpperBound.branch row8951_large_right_left_covers row8951_large_right_right_covers

def row8951_large : List Nat := row8951_large_left ++ row8951_large_right
theorem row8951_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8951 row8951_large :=
  Budget.RowUpperBound.branch row8951_large_left_covers row8951_large_right_covers

def row8951 : List Nat := row8951_large
theorem row8951_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8951 row8951 := row8951_large_covers

def row8963_small : List Nat := []
theorem row8963_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8963 row8963_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8963_block_005 : List Nat := []
def plan8963_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 13), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 17), .second, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row8963_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8963 row8963_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8963_block_005
  · decide
  · rfl

def row8963_block_006 : List Nat := [11093, 11261, 11483, 11489, 12281]
def plan8963_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .second, .first, .second, .first, (.firstFactor 17), (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .second, (.secondFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .possible, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .second, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8963_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8963 row8963_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8963_block_006
  · decide
  · rfl

def row8963_block_007 : List Nat := [12899, 13421, 14033, 14087]
def plan8963_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 17), (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, .possible, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 23), (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8963_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8963 row8963_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8963_block_007
  · decide
  · rfl

def row8963_block_008 : List Nat := [14537, 15683]
def plan8963_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .second, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 17), .first, .first, .second, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8963_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8963 row8963_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8963_block_008
  · decide
  · rfl

def row8963_block_009 : List Nat := [16901]
def plan8963_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8963_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8963 row8963_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8963_block_009
  · decide
  · rfl

def row8963_block_010 : List Nat := []
theorem row8963_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8963 row8963_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8963_block_011 : List Nat := []
theorem row8963_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8963 row8963_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8963_block_012 : List Nat := []
theorem row8963_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8963 row8963_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8963_block_013 : List Nat := []
theorem row8963_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8963 row8963_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8963_large_left_left : List Nat := row8963_small ++ row8963_block_005
theorem row8963_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8963 row8963_large_left_left :=
  Budget.RowUpperBound.branch row8963_small_covers row8963_block_005_covers

def row8963_large_left_right_right : List Nat := row8963_block_007 ++ row8963_block_008
theorem row8963_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8963 row8963_large_left_right_right :=
  Budget.RowUpperBound.branch row8963_block_007_covers row8963_block_008_covers

def row8963_large_left_right : List Nat := row8963_block_006 ++ row8963_large_left_right_right
theorem row8963_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8963 row8963_large_left_right :=
  Budget.RowUpperBound.branch row8963_block_006_covers row8963_large_left_right_right_covers

def row8963_large_left : List Nat := row8963_large_left_left ++ row8963_large_left_right
theorem row8963_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8963 row8963_large_left :=
  Budget.RowUpperBound.branch row8963_large_left_left_covers row8963_large_left_right_covers

def row8963_large_right_left : List Nat := row8963_block_009 ++ row8963_block_010
theorem row8963_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8963 row8963_large_right_left :=
  Budget.RowUpperBound.branch row8963_block_009_covers row8963_block_010_covers

def row8963_large_right_right_right : List Nat := row8963_block_012 ++ row8963_block_013
theorem row8963_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8963 row8963_large_right_right_right :=
  Budget.RowUpperBound.branch row8963_block_012_covers row8963_block_013_covers

def row8963_large_right_right : List Nat := row8963_block_011 ++ row8963_large_right_right_right
theorem row8963_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8963 row8963_large_right_right :=
  Budget.RowUpperBound.branch row8963_block_011_covers row8963_large_right_right_right_covers

def row8963_large_right : List Nat := row8963_large_right_left ++ row8963_large_right_right
theorem row8963_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8963 row8963_large_right :=
  Budget.RowUpperBound.branch row8963_large_right_left_covers row8963_large_right_right_covers

def row8963_large : List Nat := row8963_large_left ++ row8963_large_right
theorem row8963_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8963 row8963_large :=
  Budget.RowUpperBound.branch row8963_large_left_covers row8963_large_right_covers

def row8963 : List Nat := row8963_large
theorem row8963_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8963 row8963 := row8963_large_covers

end ProjectEulerSolutions.PrimePairLarge
