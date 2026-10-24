import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows080

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8179_small : List Nat := PrimePairUpper.smallRow8179
theorem row8179_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8179 row8179_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8179_covers

def row8179_block_005 : List Nat := [9181, 9463]
def plan8179_block_005 : List Budget.Choice := [(.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .second, .first, (.secondFactor 37), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .second, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 31), (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8179_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8179 row8179_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8179_block_005
  · decide
  · rfl

def row8179_block_006 : List Nat := [10657, 10663, 10831, 11527, 11587, 11821, 11833, 12037]
def plan8179_block_006 : List Budget.Choice := [.first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .possible, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .second, (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .possible, .first, (.secondFactor 7), .first, .possible, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, .possible, .second, .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 13), .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row8179_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8179 row8179_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8179_block_006
  · decide
  · rfl

def row8179_block_007 : List Nat := [12541, 12889, 13003, 13339, 13963]
def plan8179_block_007 : List Budget.Choice := [.second, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .possible, (.secondFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 37), (.firstFactor 29), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8179_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8179 row8179_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8179_block_007
  · decide
  · rfl

def row8179_block_008 : List Nat := []
def plan8179_block_008 : List Budget.Choice := [(.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 37), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8179_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8179 row8179_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8179_block_008
  · decide
  · rfl

def row8179_block_009 : List Nat := [16657, 17167, 17761]
def plan8179_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 31), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .second, (.firstFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .second, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8179_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8179 row8179_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8179_block_009
  · decide
  · rfl

def row8179_block_010 : List Nat := []
theorem row8179_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8179 row8179_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8179_block_011 : List Nat := []
theorem row8179_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8179 row8179_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8179_block_012 : List Nat := []
theorem row8179_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8179 row8179_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8179_block_013 : List Nat := []
theorem row8179_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8179 row8179_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8179_large_left_left : List Nat := row8179_small ++ row8179_block_005
theorem row8179_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8179 row8179_large_left_left :=
  Budget.RowUpperBound.branch row8179_small_covers row8179_block_005_covers

def row8179_large_left_right_right : List Nat := row8179_block_007 ++ row8179_block_008
theorem row8179_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8179 row8179_large_left_right_right :=
  Budget.RowUpperBound.branch row8179_block_007_covers row8179_block_008_covers

def row8179_large_left_right : List Nat := row8179_block_006 ++ row8179_large_left_right_right
theorem row8179_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8179 row8179_large_left_right :=
  Budget.RowUpperBound.branch row8179_block_006_covers row8179_large_left_right_right_covers

def row8179_large_left : List Nat := row8179_large_left_left ++ row8179_large_left_right
theorem row8179_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8179 row8179_large_left :=
  Budget.RowUpperBound.branch row8179_large_left_left_covers row8179_large_left_right_covers

def row8179_large_right_left : List Nat := row8179_block_009 ++ row8179_block_010
theorem row8179_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8179 row8179_large_right_left :=
  Budget.RowUpperBound.branch row8179_block_009_covers row8179_block_010_covers

def row8179_large_right_right_right : List Nat := row8179_block_012 ++ row8179_block_013
theorem row8179_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8179 row8179_large_right_right_right :=
  Budget.RowUpperBound.branch row8179_block_012_covers row8179_block_013_covers

def row8179_large_right_right : List Nat := row8179_block_011 ++ row8179_large_right_right_right
theorem row8179_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8179 row8179_large_right_right :=
  Budget.RowUpperBound.branch row8179_block_011_covers row8179_large_right_right_right_covers

def row8179_large_right : List Nat := row8179_large_right_left ++ row8179_large_right_right
theorem row8179_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8179 row8179_large_right :=
  Budget.RowUpperBound.branch row8179_large_right_left_covers row8179_large_right_right_covers

def row8179_large : List Nat := row8179_large_left ++ row8179_large_right
theorem row8179_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8179 row8179_large :=
  Budget.RowUpperBound.branch row8179_large_left_covers row8179_large_right_covers

def row8179 : List Nat := row8179_large
theorem row8179_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8179 row8179 := row8179_large_covers

def row8191_small : List Nat := PrimePairUpper.smallRow8191
theorem row8191_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8191 row8191_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8191_covers

def row8191_block_005 : List Nat := [8737, 9547, 9769, 10009]
def plan8191_block_005 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .second, .second, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 29), .first, .first, .second, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, (.secondFactor 19), .second, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8191_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8191 row8191_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8191_block_005
  · decide
  · rfl

def row8191_block_006 : List Nat := [11287, 12043]
def plan8191_block_006 : List Budget.Choice := [.first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 19), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 31), .first, (.firstFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row8191_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8191 row8191_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8191_block_006
  · decide
  · rfl

def row8191_block_007 : List Nat := [14221, 14419]
def plan8191_block_007 : List Budget.Choice := [.second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .second, .second, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 29), (.firstFactor 17), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8191_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8191 row8191_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8191_block_007
  · decide
  · rfl

def row8191_block_008 : List Nat := [14713, 14983, 15313, 15667, 16183, 16381, 16447]
def plan8191_block_008 : List Budget.Choice := [(.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .possible, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .possible, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 13), .second, .possible, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .possible, .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8191_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8191 row8191_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8191_block_008
  · decide
  · rfl

def row8191_block_009 : List Nat := [16633, 16759, 17077, 17737]
def plan8191_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 37), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 7), .second, .possible, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 17), (.firstFactor 7), .first, .second, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 37), .first, (.secondFactor 31), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .second, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8191_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8191 row8191_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8191_block_009
  · decide
  · rfl

def row8191_block_010 : List Nat := []
theorem row8191_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8191 row8191_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8191_block_011 : List Nat := []
theorem row8191_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8191 row8191_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8191_block_012 : List Nat := []
theorem row8191_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8191 row8191_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8191_block_013 : List Nat := []
theorem row8191_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8191 row8191_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8191_large_left_left : List Nat := row8191_small ++ row8191_block_005
theorem row8191_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8191 row8191_large_left_left :=
  Budget.RowUpperBound.branch row8191_small_covers row8191_block_005_covers

def row8191_large_left_right_right : List Nat := row8191_block_007 ++ row8191_block_008
theorem row8191_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8191 row8191_large_left_right_right :=
  Budget.RowUpperBound.branch row8191_block_007_covers row8191_block_008_covers

def row8191_large_left_right : List Nat := row8191_block_006 ++ row8191_large_left_right_right
theorem row8191_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8191 row8191_large_left_right :=
  Budget.RowUpperBound.branch row8191_block_006_covers row8191_large_left_right_right_covers

def row8191_large_left : List Nat := row8191_large_left_left ++ row8191_large_left_right
theorem row8191_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8191 row8191_large_left :=
  Budget.RowUpperBound.branch row8191_large_left_left_covers row8191_large_left_right_covers

def row8191_large_right_left : List Nat := row8191_block_009 ++ row8191_block_010
theorem row8191_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8191 row8191_large_right_left :=
  Budget.RowUpperBound.branch row8191_block_009_covers row8191_block_010_covers

def row8191_large_right_right_right : List Nat := row8191_block_012 ++ row8191_block_013
theorem row8191_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8191 row8191_large_right_right_right :=
  Budget.RowUpperBound.branch row8191_block_012_covers row8191_block_013_covers

def row8191_large_right_right : List Nat := row8191_block_011 ++ row8191_large_right_right_right
theorem row8191_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8191 row8191_large_right_right :=
  Budget.RowUpperBound.branch row8191_block_011_covers row8191_large_right_right_right_covers

def row8191_large_right : List Nat := row8191_large_right_left ++ row8191_large_right_right
theorem row8191_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8191 row8191_large_right :=
  Budget.RowUpperBound.branch row8191_large_right_left_covers row8191_large_right_right_covers

def row8191_large : List Nat := row8191_large_left ++ row8191_large_right
theorem row8191_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8191 row8191_large :=
  Budget.RowUpperBound.branch row8191_large_left_covers row8191_large_right_covers

def row8191 : List Nat := row8191_large
theorem row8191_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8191 row8191 := row8191_large_covers

def row8209_small : List Nat := PrimePairUpper.smallRow8209
theorem row8209_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8209 row8209_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8209_covers

def row8209_block_005 : List Nat := [8641, 8941, 9649, 9859, 10267]
def plan8209_block_005 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 19), .possible, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, (.secondFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8209_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8209 row8209_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8209_block_005
  · decide
  · rfl

def row8209_block_006 : List Nat := [12109]
def plan8209_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .second, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8209_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8209 row8209_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8209_block_006
  · decide
  · rfl

def row8209_block_007 : List Nat := [12637, 12973, 14293]
def plan8209_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 23), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 11), (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 11), .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row8209_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8209 row8209_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8209_block_007
  · decide
  · rfl

def row8209_block_008 : List Nat := [16249]
def plan8209_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .second, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .second, (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 23), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8209_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8209 row8209_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8209_block_008
  · decide
  · rfl

def row8209_block_009 : List Nat := [16633, 17191, 17341, 17491, 17569, 17599]
def plan8209_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, .second, (.secondFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 11), (.firstFactor 11), .second, .first, .first, .first, .first, .possible, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8209_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8209 row8209_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8209_block_009
  · decide
  · rfl

def row8209_block_010 : List Nat := []
theorem row8209_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8209 row8209_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8209_block_011 : List Nat := []
theorem row8209_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8209 row8209_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8209_block_012 : List Nat := []
theorem row8209_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8209 row8209_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8209_block_013 : List Nat := []
theorem row8209_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8209 row8209_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8209_large_left_left : List Nat := row8209_small ++ row8209_block_005
theorem row8209_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8209 row8209_large_left_left :=
  Budget.RowUpperBound.branch row8209_small_covers row8209_block_005_covers

def row8209_large_left_right_right : List Nat := row8209_block_007 ++ row8209_block_008
theorem row8209_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8209 row8209_large_left_right_right :=
  Budget.RowUpperBound.branch row8209_block_007_covers row8209_block_008_covers

def row8209_large_left_right : List Nat := row8209_block_006 ++ row8209_large_left_right_right
theorem row8209_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8209 row8209_large_left_right :=
  Budget.RowUpperBound.branch row8209_block_006_covers row8209_large_left_right_right_covers

def row8209_large_left : List Nat := row8209_large_left_left ++ row8209_large_left_right
theorem row8209_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8209 row8209_large_left :=
  Budget.RowUpperBound.branch row8209_large_left_left_covers row8209_large_left_right_covers

def row8209_large_right_left : List Nat := row8209_block_009 ++ row8209_block_010
theorem row8209_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8209 row8209_large_right_left :=
  Budget.RowUpperBound.branch row8209_block_009_covers row8209_block_010_covers

def row8209_large_right_right_right : List Nat := row8209_block_012 ++ row8209_block_013
theorem row8209_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8209 row8209_large_right_right_right :=
  Budget.RowUpperBound.branch row8209_block_012_covers row8209_block_013_covers

def row8209_large_right_right : List Nat := row8209_block_011 ++ row8209_large_right_right_right
theorem row8209_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8209 row8209_large_right_right :=
  Budget.RowUpperBound.branch row8209_block_011_covers row8209_large_right_right_right_covers

def row8209_large_right : List Nat := row8209_large_right_left ++ row8209_large_right_right
theorem row8209_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8209 row8209_large_right :=
  Budget.RowUpperBound.branch row8209_large_right_left_covers row8209_large_right_right_covers

def row8209_large : List Nat := row8209_large_left ++ row8209_large_right
theorem row8209_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8209 row8209_large :=
  Budget.RowUpperBound.branch row8209_large_left_covers row8209_large_right_covers

def row8209 : List Nat := row8209_large
theorem row8209_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8209 row8209 := row8209_large_covers

def row8219_small : List Nat := PrimePairUpper.smallRow8219
theorem row8219_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8219 row8219_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8219_covers

def row8219_block_005 : List Nat := [8819, 9239, 10343]
def plan8219_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 37), .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 13), (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8219_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8219 row8219_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8219_block_005
  · decide
  · rfl

def row8219_block_006 : List Nat := [11117, 11369, 12251]
def plan8219_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .second, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8219_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8219 row8219_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8219_block_006
  · decide
  · rfl

def row8219_block_007 : List Nat := [12491, 12503, 12569, 13163, 13397]
def plan8219_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .possible, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), .second, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8219_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8219 row8219_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8219_block_007
  · decide
  · rfl

def row8219_block_008 : List Nat := [14699, 15527, 15569, 16349, 16451]
def plan8219_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), .second, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8219_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8219 row8219_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8219_block_008
  · decide
  · rfl

def row8219_block_009 : List Nat := [17573]
def plan8219_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 11), (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), (.firstFactor 29), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .possible, (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8219_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8219 row8219_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8219_block_009
  · decide
  · rfl

def row8219_block_010 : List Nat := []
theorem row8219_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8219 row8219_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8219_block_011 : List Nat := []
theorem row8219_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8219 row8219_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8219_block_012 : List Nat := []
theorem row8219_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8219 row8219_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8219_block_013 : List Nat := []
theorem row8219_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8219 row8219_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8219_large_left_left : List Nat := row8219_small ++ row8219_block_005
theorem row8219_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8219 row8219_large_left_left :=
  Budget.RowUpperBound.branch row8219_small_covers row8219_block_005_covers

def row8219_large_left_right_right : List Nat := row8219_block_007 ++ row8219_block_008
theorem row8219_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8219 row8219_large_left_right_right :=
  Budget.RowUpperBound.branch row8219_block_007_covers row8219_block_008_covers

def row8219_large_left_right : List Nat := row8219_block_006 ++ row8219_large_left_right_right
theorem row8219_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8219 row8219_large_left_right :=
  Budget.RowUpperBound.branch row8219_block_006_covers row8219_large_left_right_right_covers

def row8219_large_left : List Nat := row8219_large_left_left ++ row8219_large_left_right
theorem row8219_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8219 row8219_large_left :=
  Budget.RowUpperBound.branch row8219_large_left_left_covers row8219_large_left_right_covers

def row8219_large_right_left : List Nat := row8219_block_009 ++ row8219_block_010
theorem row8219_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8219 row8219_large_right_left :=
  Budget.RowUpperBound.branch row8219_block_009_covers row8219_block_010_covers

def row8219_large_right_right_right : List Nat := row8219_block_012 ++ row8219_block_013
theorem row8219_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8219 row8219_large_right_right_right :=
  Budget.RowUpperBound.branch row8219_block_012_covers row8219_block_013_covers

def row8219_large_right_right : List Nat := row8219_block_011 ++ row8219_large_right_right_right
theorem row8219_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8219 row8219_large_right_right :=
  Budget.RowUpperBound.branch row8219_block_011_covers row8219_large_right_right_right_covers

def row8219_large_right : List Nat := row8219_large_right_left ++ row8219_large_right_right
theorem row8219_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8219 row8219_large_right :=
  Budget.RowUpperBound.branch row8219_large_right_left_covers row8219_large_right_right_covers

def row8219_large : List Nat := row8219_large_left ++ row8219_large_right
theorem row8219_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8219 row8219_large :=
  Budget.RowUpperBound.branch row8219_large_left_covers row8219_large_right_covers

def row8219 : List Nat := row8219_large
theorem row8219_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8219 row8219 := row8219_large_covers

end ProjectEulerSolutions.PrimePairLarge
