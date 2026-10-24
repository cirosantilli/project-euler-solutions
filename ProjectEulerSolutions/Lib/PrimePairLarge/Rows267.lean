import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8599_small : List Nat := []
theorem row8599_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8599 row8599_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8599_block_005 : List Nat := [8623, 9931, 10357]
def plan8599_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .possible, .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 7), .second, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), .second, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 19), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .possible, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8599_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8599 row8599_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8599_block_005
  · decide
  · rfl

def row8599_block_006 : List Nat := [12253]
def plan8599_block_006 : List Budget.Choice := [.first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 11), .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8599_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8599 row8599_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8599_block_006
  · decide
  · rfl

def row8599_block_007 : List Nat := [12457, 12577, 12841, 14431]
def plan8599_block_007 : List Budget.Choice := [(.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 31), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 13), (.firstFactor 29), .first, .possible, (.firstFactor 19), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 11), .possible, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 37), .second, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 11), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8599_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8599 row8599_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8599_block_007
  · decide
  · rfl

def row8599_block_008 : List Nat := [15307]
def plan8599_block_008 : List Budget.Choice := [(.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8599_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8599 row8599_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8599_block_008
  · decide
  · rfl

def row8599_block_009 : List Nat := [17299]
def plan8599_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .second, .first, .first, .first, (.firstFactor 29), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8599_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8599 row8599_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8599_block_009
  · decide
  · rfl

def row8599_block_010 : List Nat := []
theorem row8599_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8599 row8599_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8599_block_011 : List Nat := []
theorem row8599_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8599 row8599_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8599_block_012 : List Nat := []
theorem row8599_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8599 row8599_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8599_block_013 : List Nat := []
theorem row8599_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8599 row8599_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8599_large_left_left : List Nat := row8599_small ++ row8599_block_005
theorem row8599_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8599 row8599_large_left_left :=
  Budget.RowUpperBound.branch row8599_small_covers row8599_block_005_covers

def row8599_large_left_right_right : List Nat := row8599_block_007 ++ row8599_block_008
theorem row8599_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8599 row8599_large_left_right_right :=
  Budget.RowUpperBound.branch row8599_block_007_covers row8599_block_008_covers

def row8599_large_left_right : List Nat := row8599_block_006 ++ row8599_large_left_right_right
theorem row8599_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8599 row8599_large_left_right :=
  Budget.RowUpperBound.branch row8599_block_006_covers row8599_large_left_right_right_covers

def row8599_large_left : List Nat := row8599_large_left_left ++ row8599_large_left_right
theorem row8599_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8599 row8599_large_left :=
  Budget.RowUpperBound.branch row8599_large_left_left_covers row8599_large_left_right_covers

def row8599_large_right_left : List Nat := row8599_block_009 ++ row8599_block_010
theorem row8599_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8599 row8599_large_right_left :=
  Budget.RowUpperBound.branch row8599_block_009_covers row8599_block_010_covers

def row8599_large_right_right_right : List Nat := row8599_block_012 ++ row8599_block_013
theorem row8599_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8599 row8599_large_right_right_right :=
  Budget.RowUpperBound.branch row8599_block_012_covers row8599_block_013_covers

def row8599_large_right_right : List Nat := row8599_block_011 ++ row8599_large_right_right_right
theorem row8599_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8599 row8599_large_right_right :=
  Budget.RowUpperBound.branch row8599_block_011_covers row8599_large_right_right_right_covers

def row8599_large_right : List Nat := row8599_large_right_left ++ row8599_large_right_right
theorem row8599_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8599 row8599_large_right :=
  Budget.RowUpperBound.branch row8599_large_right_left_covers row8599_large_right_right_covers

def row8599_large : List Nat := row8599_large_left ++ row8599_large_right
theorem row8599_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8599 row8599_large :=
  Budget.RowUpperBound.branch row8599_large_left_covers row8599_large_right_covers

def row8599 : List Nat := row8599_large
theorem row8599_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8599 row8599 := row8599_large_covers

def row8609_small : List Nat := []
theorem row8609_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8609 row8609_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8609_block_005 : List Nat := [9491, 9743]
def plan8609_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), (.secondFactor 29), .second, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .possible, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .possible, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), .second, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8609_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8609 row8609_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8609_block_005
  · decide
  · rfl

def row8609_block_006 : List Nat := [10601, 10781, 10937, 11927, 12347]
def plan8609_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, (.secondFactor 31), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8609_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8609 row8609_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8609_block_006
  · decide
  · rfl

def row8609_block_007 : List Nat := [12809, 12893, 14207]
def plan8609_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 29), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8609_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8609 row8609_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8609_block_007
  · decide
  · rfl

def row8609_block_008 : List Nat := [14543, 15527]
def plan8609_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .second, .second, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 31), .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 37), (.secondFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8609_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8609 row8609_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8609_block_008
  · decide
  · rfl

def row8609_block_009 : List Nat := [16547]
def plan8609_block_009 : List Budget.Choice := [.first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8609_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8609 row8609_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8609_block_009
  · decide
  · rfl

def row8609_block_010 : List Nat := []
theorem row8609_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8609 row8609_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8609_block_011 : List Nat := []
theorem row8609_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8609 row8609_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8609_block_012 : List Nat := []
theorem row8609_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8609 row8609_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8609_block_013 : List Nat := []
theorem row8609_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8609 row8609_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8609_large_left_left : List Nat := row8609_small ++ row8609_block_005
theorem row8609_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8609 row8609_large_left_left :=
  Budget.RowUpperBound.branch row8609_small_covers row8609_block_005_covers

def row8609_large_left_right_right : List Nat := row8609_block_007 ++ row8609_block_008
theorem row8609_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8609 row8609_large_left_right_right :=
  Budget.RowUpperBound.branch row8609_block_007_covers row8609_block_008_covers

def row8609_large_left_right : List Nat := row8609_block_006 ++ row8609_large_left_right_right
theorem row8609_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8609 row8609_large_left_right :=
  Budget.RowUpperBound.branch row8609_block_006_covers row8609_large_left_right_right_covers

def row8609_large_left : List Nat := row8609_large_left_left ++ row8609_large_left_right
theorem row8609_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8609 row8609_large_left :=
  Budget.RowUpperBound.branch row8609_large_left_left_covers row8609_large_left_right_covers

def row8609_large_right_left : List Nat := row8609_block_009 ++ row8609_block_010
theorem row8609_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8609 row8609_large_right_left :=
  Budget.RowUpperBound.branch row8609_block_009_covers row8609_block_010_covers

def row8609_large_right_right_right : List Nat := row8609_block_012 ++ row8609_block_013
theorem row8609_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8609 row8609_large_right_right_right :=
  Budget.RowUpperBound.branch row8609_block_012_covers row8609_block_013_covers

def row8609_large_right_right : List Nat := row8609_block_011 ++ row8609_large_right_right_right
theorem row8609_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8609 row8609_large_right_right :=
  Budget.RowUpperBound.branch row8609_block_011_covers row8609_large_right_right_right_covers

def row8609_large_right : List Nat := row8609_large_right_left ++ row8609_large_right_right
theorem row8609_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8609 row8609_large_right :=
  Budget.RowUpperBound.branch row8609_large_right_left_covers row8609_large_right_right_covers

def row8609_large : List Nat := row8609_large_left ++ row8609_large_right
theorem row8609_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8609 row8609_large :=
  Budget.RowUpperBound.branch row8609_large_left_covers row8609_large_right_covers

def row8609 : List Nat := row8609_large
theorem row8609_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8609 row8609 := row8609_large_covers

def row8623_small : List Nat := []
theorem row8623_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8623 row8623_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8623_block_005 : List Nat := [8863, 9091, 9619, 10093, 10333]
def plan8623_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .possible, (.secondFactor 13), (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .second, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 23), (.firstFactor 11), .second, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, .possible, .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8623_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8623 row8623_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8623_block_005
  · decide
  · rfl

def row8623_block_006 : List Nat := [10597, 10723, 10909, 11119]
def plan8623_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .second, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 29), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8623_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8623 row8623_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8623_block_006
  · decide
  · rfl

def row8623_block_007 : List Nat := [12421, 13147, 13513, 13789, 14437]
def plan8623_block_007 : List Budget.Choice := [.possible, .second, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 11), .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 37), (.firstFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, .possible, .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 19), .second]
set_option maxRecDepth 2048 in
theorem row8623_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8623 row8623_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8623_block_007
  · decide
  · rfl

def row8623_block_008 : List Nat := [15217, 15541, 16033]
def plan8623_block_008 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .second, (.secondFactor 7), .possible, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), (.secondFactor 31), (.secondFactor 11), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8623_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8623 row8623_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8623_block_008
  · decide
  · rfl

def row8623_block_009 : List Nat := [17041]
def plan8623_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 13), .second, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .second, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8623_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8623 row8623_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8623_block_009
  · decide
  · rfl

def row8623_block_010 : List Nat := []
theorem row8623_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8623 row8623_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8623_block_011 : List Nat := []
theorem row8623_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8623 row8623_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8623_block_012 : List Nat := []
theorem row8623_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8623 row8623_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8623_block_013 : List Nat := []
theorem row8623_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8623 row8623_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8623_large_left_left : List Nat := row8623_small ++ row8623_block_005
theorem row8623_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8623 row8623_large_left_left :=
  Budget.RowUpperBound.branch row8623_small_covers row8623_block_005_covers

def row8623_large_left_right_right : List Nat := row8623_block_007 ++ row8623_block_008
theorem row8623_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8623 row8623_large_left_right_right :=
  Budget.RowUpperBound.branch row8623_block_007_covers row8623_block_008_covers

def row8623_large_left_right : List Nat := row8623_block_006 ++ row8623_large_left_right_right
theorem row8623_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8623 row8623_large_left_right :=
  Budget.RowUpperBound.branch row8623_block_006_covers row8623_large_left_right_right_covers

def row8623_large_left : List Nat := row8623_large_left_left ++ row8623_large_left_right
theorem row8623_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8623 row8623_large_left :=
  Budget.RowUpperBound.branch row8623_large_left_left_covers row8623_large_left_right_covers

def row8623_large_right_left : List Nat := row8623_block_009 ++ row8623_block_010
theorem row8623_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8623 row8623_large_right_left :=
  Budget.RowUpperBound.branch row8623_block_009_covers row8623_block_010_covers

def row8623_large_right_right_right : List Nat := row8623_block_012 ++ row8623_block_013
theorem row8623_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8623 row8623_large_right_right_right :=
  Budget.RowUpperBound.branch row8623_block_012_covers row8623_block_013_covers

def row8623_large_right_right : List Nat := row8623_block_011 ++ row8623_large_right_right_right
theorem row8623_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8623 row8623_large_right_right :=
  Budget.RowUpperBound.branch row8623_block_011_covers row8623_large_right_right_right_covers

def row8623_large_right : List Nat := row8623_large_right_left ++ row8623_large_right_right
theorem row8623_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8623 row8623_large_right :=
  Budget.RowUpperBound.branch row8623_large_right_left_covers row8623_large_right_right_covers

def row8623_large : List Nat := row8623_large_left ++ row8623_large_right
theorem row8623_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8623 row8623_large :=
  Budget.RowUpperBound.branch row8623_large_left_covers row8623_large_right_covers

def row8623 : List Nat := row8623_large
theorem row8623_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8623 row8623 := row8623_large_covers

def row8627_small : List Nat := []
theorem row8627_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8627 row8627_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8627_block_005 : List Nat := [8861, 9173, 9257, 9533, 10313]
def plan8627_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .possible, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 37), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 11), .second, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 19), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8627_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8627 row8627_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8627_block_005
  · decide
  · rfl

def row8627_block_006 : List Nat := [11939]
def plan8627_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8627_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8627 row8627_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8627_block_006
  · decide
  · rfl

def row8627_block_007 : List Nat := [13121]
def plan8627_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 23), (.firstFactor 29), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .second, (.firstFactor 31), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .possible, (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8627_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8627 row8627_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8627_block_007
  · decide
  · rfl

def row8627_block_008 : List Nat := [14543, 14639, 15101, 15137, 15263, 15731]
def plan8627_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .second, (.firstFactor 19), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .second, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .possible, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8627_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8627 row8627_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8627_block_008
  · decide
  · rfl

def row8627_block_009 : List Nat := [16931, 17159]
def plan8627_block_009 : List Budget.Choice := [.first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .possible, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .possible, .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 23), (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8627_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8627 row8627_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8627_block_009
  · decide
  · rfl

def row8627_block_010 : List Nat := []
theorem row8627_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8627 row8627_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8627_block_011 : List Nat := []
theorem row8627_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8627 row8627_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8627_block_012 : List Nat := []
theorem row8627_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8627 row8627_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8627_block_013 : List Nat := []
theorem row8627_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8627 row8627_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8627_large_left_left : List Nat := row8627_small ++ row8627_block_005
theorem row8627_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8627 row8627_large_left_left :=
  Budget.RowUpperBound.branch row8627_small_covers row8627_block_005_covers

def row8627_large_left_right_right : List Nat := row8627_block_007 ++ row8627_block_008
theorem row8627_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8627 row8627_large_left_right_right :=
  Budget.RowUpperBound.branch row8627_block_007_covers row8627_block_008_covers

def row8627_large_left_right : List Nat := row8627_block_006 ++ row8627_large_left_right_right
theorem row8627_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8627 row8627_large_left_right :=
  Budget.RowUpperBound.branch row8627_block_006_covers row8627_large_left_right_right_covers

def row8627_large_left : List Nat := row8627_large_left_left ++ row8627_large_left_right
theorem row8627_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8627 row8627_large_left :=
  Budget.RowUpperBound.branch row8627_large_left_left_covers row8627_large_left_right_covers

def row8627_large_right_left : List Nat := row8627_block_009 ++ row8627_block_010
theorem row8627_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8627 row8627_large_right_left :=
  Budget.RowUpperBound.branch row8627_block_009_covers row8627_block_010_covers

def row8627_large_right_right_right : List Nat := row8627_block_012 ++ row8627_block_013
theorem row8627_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8627 row8627_large_right_right_right :=
  Budget.RowUpperBound.branch row8627_block_012_covers row8627_block_013_covers

def row8627_large_right_right : List Nat := row8627_block_011 ++ row8627_large_right_right_right
theorem row8627_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8627 row8627_large_right_right :=
  Budget.RowUpperBound.branch row8627_block_011_covers row8627_large_right_right_right_covers

def row8627_large_right : List Nat := row8627_large_right_left ++ row8627_large_right_right
theorem row8627_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8627 row8627_large_right :=
  Budget.RowUpperBound.branch row8627_large_right_left_covers row8627_large_right_right_covers

def row8627_large : List Nat := row8627_large_left ++ row8627_large_right
theorem row8627_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8627 row8627_large :=
  Budget.RowUpperBound.branch row8627_large_left_covers row8627_large_right_covers

def row8627 : List Nat := row8627_large
theorem row8627_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8627 row8627 := row8627_large_covers

end ProjectEulerSolutions.PrimePairLarge
