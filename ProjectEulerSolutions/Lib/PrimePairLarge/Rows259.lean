import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows081

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8287_small : List Nat := PrimePairUpper.smallRow8287
theorem row8287_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8287 row8287_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8287_covers

def row8287_block_005 : List Nat := [9133, 9721, 9859, 10357]
def plan8287_block_005 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 17), .possible, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8287_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8287 row8287_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8287_block_005
  · decide
  · rfl

def row8287_block_006 : List Nat := [11173, 12253]
def plan8287_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .second, .second, (.firstFactor 7), .first, (.secondFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8287_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8287 row8287_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8287_block_006
  · decide
  · rfl

def row8287_block_007 : List Nat := [12433, 13597, 14347]
def plan8287_block_007 : List Budget.Choice := [(.firstFactor 37), .possible, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 19), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .second, .first, (.firstFactor 23), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8287_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8287 row8287_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8287_block_007
  · decide
  · rfl

def row8287_block_008 : List Nat := [14713]
def plan8287_block_008 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8287_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8287 row8287_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8287_block_008
  · decide
  · rfl

def row8287_block_009 : List Nat := [16741, 17713]
def plan8287_block_009 : List Budget.Choice := [(.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .possible, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 19), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .possible, .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8287_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8287 row8287_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8287_block_009
  · decide
  · rfl

def row8287_block_010 : List Nat := []
theorem row8287_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8287 row8287_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8287_block_011 : List Nat := []
theorem row8287_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8287 row8287_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8287_block_012 : List Nat := []
theorem row8287_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8287 row8287_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8287_block_013 : List Nat := []
theorem row8287_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8287 row8287_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8287_large_left_left : List Nat := row8287_small ++ row8287_block_005
theorem row8287_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8287 row8287_large_left_left :=
  Budget.RowUpperBound.branch row8287_small_covers row8287_block_005_covers

def row8287_large_left_right_right : List Nat := row8287_block_007 ++ row8287_block_008
theorem row8287_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8287 row8287_large_left_right_right :=
  Budget.RowUpperBound.branch row8287_block_007_covers row8287_block_008_covers

def row8287_large_left_right : List Nat := row8287_block_006 ++ row8287_large_left_right_right
theorem row8287_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8287 row8287_large_left_right :=
  Budget.RowUpperBound.branch row8287_block_006_covers row8287_large_left_right_right_covers

def row8287_large_left : List Nat := row8287_large_left_left ++ row8287_large_left_right
theorem row8287_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8287 row8287_large_left :=
  Budget.RowUpperBound.branch row8287_large_left_left_covers row8287_large_left_right_covers

def row8287_large_right_left : List Nat := row8287_block_009 ++ row8287_block_010
theorem row8287_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8287 row8287_large_right_left :=
  Budget.RowUpperBound.branch row8287_block_009_covers row8287_block_010_covers

def row8287_large_right_right_right : List Nat := row8287_block_012 ++ row8287_block_013
theorem row8287_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8287 row8287_large_right_right_right :=
  Budget.RowUpperBound.branch row8287_block_012_covers row8287_block_013_covers

def row8287_large_right_right : List Nat := row8287_block_011 ++ row8287_large_right_right_right
theorem row8287_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8287 row8287_large_right_right :=
  Budget.RowUpperBound.branch row8287_block_011_covers row8287_large_right_right_right_covers

def row8287_large_right : List Nat := row8287_large_right_left ++ row8287_large_right_right
theorem row8287_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8287 row8287_large_right :=
  Budget.RowUpperBound.branch row8287_large_right_left_covers row8287_large_right_right_covers

def row8287_large : List Nat := row8287_large_left ++ row8287_large_right
theorem row8287_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8287 row8287_large :=
  Budget.RowUpperBound.branch row8287_large_left_covers row8287_large_right_covers

def row8287 : List Nat := row8287_large
theorem row8287_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8287 row8287 := row8287_large_covers

def row8291_small : List Nat := PrimePairUpper.smallRow8291
theorem row8291_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8291 row8291_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8291_covers

def row8291_block_005 : List Nat := [8573, 8747, 8861, 9533, 9587, 9887, 10259]
def plan8291_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 37), .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 13), .possible, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 19), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .second, (.secondFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .possible, .first, .first, (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.secondFactor 31), (.secondFactor 7), (.secondFactor 11), .second, .first, (.firstFactor 7), .second, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8291_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8291 row8291_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8291_block_005
  · decide
  · rfl

def row8291_block_006 : List Nat := [10589, 11057, 11279, 11351, 11903, 12071, 12161]
def plan8291_block_006 : List Budget.Choice := [(.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, (.secondFactor 29), .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 31), (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 17), .possible, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .possible, .first, .first, (.secondFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row8291_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8291 row8291_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8291_block_006
  · decide
  · rfl

def row8291_block_007 : List Nat := [12953, 12959, 13457, 13829, 13877]
def plan8291_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .second, (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .possible, .possible, .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .possible, .second, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 29), .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8291_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8291 row8291_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8291_block_007
  · decide
  · rfl

def row8291_block_008 : List Nat := [14591, 15473, 15629, 15971, 16229]
def plan8291_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .possible, .second, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 11), (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 17), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8291_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8291 row8291_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8291_block_008
  · decide
  · rfl

def row8291_block_009 : List Nat := []
def plan8291_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8291_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8291 row8291_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8291_block_009
  · decide
  · rfl

def row8291_block_010 : List Nat := []
theorem row8291_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8291 row8291_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8291_block_011 : List Nat := []
theorem row8291_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8291 row8291_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8291_block_012 : List Nat := []
theorem row8291_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8291 row8291_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8291_block_013 : List Nat := []
theorem row8291_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8291 row8291_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8291_large_left_left : List Nat := row8291_small ++ row8291_block_005
theorem row8291_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8291 row8291_large_left_left :=
  Budget.RowUpperBound.branch row8291_small_covers row8291_block_005_covers

def row8291_large_left_right_right : List Nat := row8291_block_007 ++ row8291_block_008
theorem row8291_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8291 row8291_large_left_right_right :=
  Budget.RowUpperBound.branch row8291_block_007_covers row8291_block_008_covers

def row8291_large_left_right : List Nat := row8291_block_006 ++ row8291_large_left_right_right
theorem row8291_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8291 row8291_large_left_right :=
  Budget.RowUpperBound.branch row8291_block_006_covers row8291_large_left_right_right_covers

def row8291_large_left : List Nat := row8291_large_left_left ++ row8291_large_left_right
theorem row8291_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8291 row8291_large_left :=
  Budget.RowUpperBound.branch row8291_large_left_left_covers row8291_large_left_right_covers

def row8291_large_right_left : List Nat := row8291_block_009 ++ row8291_block_010
theorem row8291_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8291 row8291_large_right_left :=
  Budget.RowUpperBound.branch row8291_block_009_covers row8291_block_010_covers

def row8291_large_right_right_right : List Nat := row8291_block_012 ++ row8291_block_013
theorem row8291_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8291 row8291_large_right_right_right :=
  Budget.RowUpperBound.branch row8291_block_012_covers row8291_block_013_covers

def row8291_large_right_right : List Nat := row8291_block_011 ++ row8291_large_right_right_right
theorem row8291_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8291 row8291_large_right_right :=
  Budget.RowUpperBound.branch row8291_block_011_covers row8291_large_right_right_right_covers

def row8291_large_right : List Nat := row8291_large_right_left ++ row8291_large_right_right
theorem row8291_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8291 row8291_large_right :=
  Budget.RowUpperBound.branch row8291_large_right_left_covers row8291_large_right_right_covers

def row8291_large : List Nat := row8291_large_left ++ row8291_large_right
theorem row8291_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8291 row8291_large :=
  Budget.RowUpperBound.branch row8291_large_left_covers row8291_large_right_covers

def row8291 : List Nat := row8291_large
theorem row8291_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8291 row8291 := row8291_large_covers

def row8293_small : List Nat := PrimePairUpper.smallRow8293
theorem row8293_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8293 row8293_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8293_covers

def row8293_block_005 : List Nat := [8389, 8431, 8689, 9133, 9697, 9949, 10039]
def plan8293_block_005 : List Budget.Choice := [.possible, .second, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 23), .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .second, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 23), .possible, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .second, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 11), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row8293_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8293 row8293_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8293_block_005
  · decide
  · rfl

def row8293_block_006 : List Nat := [10501, 11467, 11719, 12157]
def plan8293_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 29), .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 23), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8293_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8293 row8293_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8293_block_006
  · decide
  · rfl

def row8293_block_007 : List Nat := [12433, 12973, 13411]
def plan8293_block_007 : List Budget.Choice := [(.firstFactor 7), .possible, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .second, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row8293_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8293 row8293_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8293_block_007
  · decide
  · rfl

def row8293_block_008 : List Nat := [15121, 15307, 15541]
def plan8293_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .second, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .second, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8293_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8293 row8293_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8293_block_008
  · decide
  · rfl

def row8293_block_009 : List Nat := [17389]
def plan8293_block_009 : List Budget.Choice := [(.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .second, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, .second, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8293_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8293 row8293_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8293_block_009
  · decide
  · rfl

def row8293_block_010 : List Nat := []
theorem row8293_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8293 row8293_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8293_block_011 : List Nat := []
theorem row8293_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8293 row8293_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8293_block_012 : List Nat := []
theorem row8293_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8293 row8293_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8293_block_013 : List Nat := []
theorem row8293_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8293 row8293_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8293_large_left_left : List Nat := row8293_small ++ row8293_block_005
theorem row8293_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8293 row8293_large_left_left :=
  Budget.RowUpperBound.branch row8293_small_covers row8293_block_005_covers

def row8293_large_left_right_right : List Nat := row8293_block_007 ++ row8293_block_008
theorem row8293_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8293 row8293_large_left_right_right :=
  Budget.RowUpperBound.branch row8293_block_007_covers row8293_block_008_covers

def row8293_large_left_right : List Nat := row8293_block_006 ++ row8293_large_left_right_right
theorem row8293_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8293 row8293_large_left_right :=
  Budget.RowUpperBound.branch row8293_block_006_covers row8293_large_left_right_right_covers

def row8293_large_left : List Nat := row8293_large_left_left ++ row8293_large_left_right
theorem row8293_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8293 row8293_large_left :=
  Budget.RowUpperBound.branch row8293_large_left_left_covers row8293_large_left_right_covers

def row8293_large_right_left : List Nat := row8293_block_009 ++ row8293_block_010
theorem row8293_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8293 row8293_large_right_left :=
  Budget.RowUpperBound.branch row8293_block_009_covers row8293_block_010_covers

def row8293_large_right_right_right : List Nat := row8293_block_012 ++ row8293_block_013
theorem row8293_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8293 row8293_large_right_right_right :=
  Budget.RowUpperBound.branch row8293_block_012_covers row8293_block_013_covers

def row8293_large_right_right : List Nat := row8293_block_011 ++ row8293_large_right_right_right
theorem row8293_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8293 row8293_large_right_right :=
  Budget.RowUpperBound.branch row8293_block_011_covers row8293_large_right_right_right_covers

def row8293_large_right : List Nat := row8293_large_right_left ++ row8293_large_right_right
theorem row8293_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8293 row8293_large_right :=
  Budget.RowUpperBound.branch row8293_large_right_left_covers row8293_large_right_right_covers

def row8293_large : List Nat := row8293_large_left ++ row8293_large_right
theorem row8293_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8293 row8293_large :=
  Budget.RowUpperBound.branch row8293_large_left_covers row8293_large_right_covers

def row8293 : List Nat := row8293_large
theorem row8293_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8293 row8293 := row8293_large_covers

def row8297_small : List Nat := PrimePairUpper.smallRow8297
theorem row8297_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8297 row8297_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8297_covers

def row8297_block_005 : List Nat := [9041, 9059, 9161]
def plan8297_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 11), .second, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 29), (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 31), (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8297_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8297 row8297_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8297_block_005
  · decide
  · rfl

def row8297_block_006 : List Nat := [10457, 10631, 10709, 11423]
def plan8297_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .possible, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .second, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8297_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8297 row8297_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8297_block_006
  · decide
  · rfl

def row8297_block_007 : List Nat := [12899, 13007, 14387]
def plan8297_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), (.secondFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row8297_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8297 row8297_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8297_block_007
  · decide
  · rfl

def row8297_block_008 : List Nat := [15077, 15233, 15329, 15773, 16193, 16451]
def plan8297_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .possible, .first, .second, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .second, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .possible, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8297_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8297 row8297_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8297_block_008
  · decide
  · rfl

def row8297_block_009 : List Nat := [16661]
def plan8297_block_009 : List Budget.Choice := [.first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 17), .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8297_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8297 row8297_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8297_block_009
  · decide
  · rfl

def row8297_block_010 : List Nat := []
theorem row8297_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8297 row8297_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8297_block_011 : List Nat := []
theorem row8297_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8297 row8297_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8297_block_012 : List Nat := []
theorem row8297_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8297 row8297_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8297_block_013 : List Nat := []
theorem row8297_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8297 row8297_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8297_large_left_left : List Nat := row8297_small ++ row8297_block_005
theorem row8297_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8297 row8297_large_left_left :=
  Budget.RowUpperBound.branch row8297_small_covers row8297_block_005_covers

def row8297_large_left_right_right : List Nat := row8297_block_007 ++ row8297_block_008
theorem row8297_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8297 row8297_large_left_right_right :=
  Budget.RowUpperBound.branch row8297_block_007_covers row8297_block_008_covers

def row8297_large_left_right : List Nat := row8297_block_006 ++ row8297_large_left_right_right
theorem row8297_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8297 row8297_large_left_right :=
  Budget.RowUpperBound.branch row8297_block_006_covers row8297_large_left_right_right_covers

def row8297_large_left : List Nat := row8297_large_left_left ++ row8297_large_left_right
theorem row8297_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8297 row8297_large_left :=
  Budget.RowUpperBound.branch row8297_large_left_left_covers row8297_large_left_right_covers

def row8297_large_right_left : List Nat := row8297_block_009 ++ row8297_block_010
theorem row8297_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8297 row8297_large_right_left :=
  Budget.RowUpperBound.branch row8297_block_009_covers row8297_block_010_covers

def row8297_large_right_right_right : List Nat := row8297_block_012 ++ row8297_block_013
theorem row8297_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8297 row8297_large_right_right_right :=
  Budget.RowUpperBound.branch row8297_block_012_covers row8297_block_013_covers

def row8297_large_right_right : List Nat := row8297_block_011 ++ row8297_large_right_right_right
theorem row8297_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8297 row8297_large_right_right :=
  Budget.RowUpperBound.branch row8297_block_011_covers row8297_large_right_right_right_covers

def row8297_large_right : List Nat := row8297_large_right_left ++ row8297_large_right_right
theorem row8297_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8297 row8297_large_right :=
  Budget.RowUpperBound.branch row8297_large_right_left_covers row8297_large_right_right_covers

def row8297_large : List Nat := row8297_large_left ++ row8297_large_right
theorem row8297_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8297 row8297_large :=
  Budget.RowUpperBound.branch row8297_large_left_covers row8297_large_right_covers

def row8297 : List Nat := row8297_large
theorem row8297_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8297 row8297 := row8297_large_covers

end ProjectEulerSolutions.PrimePairLarge
