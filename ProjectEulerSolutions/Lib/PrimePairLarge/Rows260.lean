import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows082

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8311_small : List Nat := PrimePairUpper.smallRow8311
theorem row8311_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8311 row8311_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8311_covers

def row8311_block_005 : List Nat := [9349]
def plan8311_block_005 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 37), .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .second, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, (.firstFactor 11), (.firstFactor 17), .second, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, .possible, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8311_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8311 row8311_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8311_block_005
  · decide
  · rfl

def row8311_block_006 : List Nat := [10459, 11299, 12109]
def plan8311_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .second, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .second, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8311_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8311 row8311_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8311_block_006
  · decide
  · rfl

def row8311_block_007 : List Nat := [12973, 13033, 13147, 14281]
def plan8311_block_007 : List Budget.Choice := [(.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row8311_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8311 row8311_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8311_block_007
  · decide
  · rfl

def row8311_block_008 : List Nat := [14551, 14653, 14737, 14923, 15013, 15073, 15217]
def plan8311_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 23), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 17), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .possible, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8311_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8311 row8311_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8311_block_008
  · decide
  · rfl

def row8311_block_009 : List Nat := [17239]
def plan8311_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.firstFactor 19), .second, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, (.secondFactor 19), .first, (.firstFactor 29), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 13), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8311_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8311 row8311_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8311_block_009
  · decide
  · rfl

def row8311_block_010 : List Nat := []
theorem row8311_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8311 row8311_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8311_block_011 : List Nat := []
theorem row8311_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8311 row8311_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8311_block_012 : List Nat := []
theorem row8311_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8311 row8311_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8311_block_013 : List Nat := []
theorem row8311_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8311 row8311_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8311_large_left_left : List Nat := row8311_small ++ row8311_block_005
theorem row8311_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8311 row8311_large_left_left :=
  Budget.RowUpperBound.branch row8311_small_covers row8311_block_005_covers

def row8311_large_left_right_right : List Nat := row8311_block_007 ++ row8311_block_008
theorem row8311_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8311 row8311_large_left_right_right :=
  Budget.RowUpperBound.branch row8311_block_007_covers row8311_block_008_covers

def row8311_large_left_right : List Nat := row8311_block_006 ++ row8311_large_left_right_right
theorem row8311_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8311 row8311_large_left_right :=
  Budget.RowUpperBound.branch row8311_block_006_covers row8311_large_left_right_right_covers

def row8311_large_left : List Nat := row8311_large_left_left ++ row8311_large_left_right
theorem row8311_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8311 row8311_large_left :=
  Budget.RowUpperBound.branch row8311_large_left_left_covers row8311_large_left_right_covers

def row8311_large_right_left : List Nat := row8311_block_009 ++ row8311_block_010
theorem row8311_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8311 row8311_large_right_left :=
  Budget.RowUpperBound.branch row8311_block_009_covers row8311_block_010_covers

def row8311_large_right_right_right : List Nat := row8311_block_012 ++ row8311_block_013
theorem row8311_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8311 row8311_large_right_right_right :=
  Budget.RowUpperBound.branch row8311_block_012_covers row8311_block_013_covers

def row8311_large_right_right : List Nat := row8311_block_011 ++ row8311_large_right_right_right
theorem row8311_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8311 row8311_large_right_right :=
  Budget.RowUpperBound.branch row8311_block_011_covers row8311_large_right_right_right_covers

def row8311_large_right : List Nat := row8311_large_right_left ++ row8311_large_right_right
theorem row8311_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8311 row8311_large_right :=
  Budget.RowUpperBound.branch row8311_large_right_left_covers row8311_large_right_right_covers

def row8311_large : List Nat := row8311_large_left ++ row8311_large_right
theorem row8311_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8311 row8311_large :=
  Budget.RowUpperBound.branch row8311_large_left_covers row8311_large_right_covers

def row8311 : List Nat := row8311_large
theorem row8311_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8311 row8311 := row8311_large_covers

def row8317_small : List Nat := PrimePairUpper.smallRow8317
theorem row8317_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8317 row8317_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8317_covers

def row8317_block_005 : List Nat := [8527, 9781, 10357]
def plan8317_block_005 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .possible, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 37), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .possible, (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8317_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8317 row8317_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8317_block_005
  · decide
  · rfl

def row8317_block_006 : List Nat := [11887]
def plan8317_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 23), (.secondFactor 11), (.firstFactor 19), .first, .second, .first, .second, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 31), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 37), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8317_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8317 row8317_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8317_block_006
  · decide
  · rfl

def row8317_block_007 : List Nat := [12907, 13873]
def plan8317_block_007 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .second, (.secondFactor 19), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, .second, .first, (.firstFactor 23), .second, .first, (.secondFactor 17), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8317_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8317 row8317_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8317_block_007
  · decide
  · rfl

def row8317_block_008 : List Nat := [14731, 15427]
def plan8317_block_008 : List Budget.Choice := [(.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 17), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8317_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8317 row8317_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8317_block_008
  · decide
  · rfl

def row8317_block_009 : List Nat := [16759]
def plan8317_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .second, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), (.firstFactor 37), .possible, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 17), (.secondFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8317_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8317 row8317_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8317_block_009
  · decide
  · rfl

def row8317_block_010 : List Nat := []
theorem row8317_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8317 row8317_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8317_block_011 : List Nat := []
theorem row8317_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8317 row8317_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8317_block_012 : List Nat := []
theorem row8317_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8317 row8317_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8317_block_013 : List Nat := []
theorem row8317_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8317 row8317_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8317_large_left_left : List Nat := row8317_small ++ row8317_block_005
theorem row8317_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8317 row8317_large_left_left :=
  Budget.RowUpperBound.branch row8317_small_covers row8317_block_005_covers

def row8317_large_left_right_right : List Nat := row8317_block_007 ++ row8317_block_008
theorem row8317_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8317 row8317_large_left_right_right :=
  Budget.RowUpperBound.branch row8317_block_007_covers row8317_block_008_covers

def row8317_large_left_right : List Nat := row8317_block_006 ++ row8317_large_left_right_right
theorem row8317_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8317 row8317_large_left_right :=
  Budget.RowUpperBound.branch row8317_block_006_covers row8317_large_left_right_right_covers

def row8317_large_left : List Nat := row8317_large_left_left ++ row8317_large_left_right
theorem row8317_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8317 row8317_large_left :=
  Budget.RowUpperBound.branch row8317_large_left_left_covers row8317_large_left_right_covers

def row8317_large_right_left : List Nat := row8317_block_009 ++ row8317_block_010
theorem row8317_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8317 row8317_large_right_left :=
  Budget.RowUpperBound.branch row8317_block_009_covers row8317_block_010_covers

def row8317_large_right_right_right : List Nat := row8317_block_012 ++ row8317_block_013
theorem row8317_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8317 row8317_large_right_right_right :=
  Budget.RowUpperBound.branch row8317_block_012_covers row8317_block_013_covers

def row8317_large_right_right : List Nat := row8317_block_011 ++ row8317_large_right_right_right
theorem row8317_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8317 row8317_large_right_right :=
  Budget.RowUpperBound.branch row8317_block_011_covers row8317_large_right_right_right_covers

def row8317_large_right : List Nat := row8317_large_right_left ++ row8317_large_right_right
theorem row8317_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8317 row8317_large_right :=
  Budget.RowUpperBound.branch row8317_large_right_left_covers row8317_large_right_right_covers

def row8317_large : List Nat := row8317_large_left ++ row8317_large_right
theorem row8317_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8317 row8317_large :=
  Budget.RowUpperBound.branch row8317_large_left_covers row8317_large_right_covers

def row8317 : List Nat := row8317_large
theorem row8317_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8317 row8317 := row8317_large_covers

def row8329_small : List Nat := PrimePairUpper.smallRow8329
theorem row8329_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8329 row8329_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8329_covers

def row8329_block_005 : List Nat := [8461, 8539, 10303, 10333]
def plan8329_block_005 : List Budget.Choice := [.second, (.firstFactor 17), .first, .first, .first, (.secondFactor 37), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 19), (.secondFactor 7), (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 23), .second, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .possible, .first, .first, (.secondFactor 19), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8329_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8329 row8329_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8329_block_005
  · decide
  · rfl

def row8329_block_006 : List Nat := [10627, 11353, 12037]
def plan8329_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, .first, (.firstFactor 11), .second, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 37), .first, .possible, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, (.secondFactor 19), (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8329_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8329 row8329_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8329_block_006
  · decide
  · rfl

def row8329_block_007 : List Nat := [13399, 14431]
def plan8329_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .second, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 19), .first, .possible, (.firstFactor 37), .first, (.secondFactor 29), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8329_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8329 row8329_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8329_block_007
  · decide
  · rfl

def row8329_block_008 : List Nat := [15427, 16033]
def plan8329_block_008 : List Budget.Choice := [(.firstFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 29), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 31), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 37), .first, .first, .possible, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 29), .second, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8329_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8329 row8329_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8329_block_008
  · decide
  · rfl

def row8329_block_009 : List Nat := []
def plan8329_block_009 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8329_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8329 row8329_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8329_block_009
  · decide
  · rfl

def row8329_block_010 : List Nat := []
theorem row8329_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8329 row8329_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8329_block_011 : List Nat := []
theorem row8329_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8329 row8329_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8329_block_012 : List Nat := []
theorem row8329_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8329 row8329_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8329_block_013 : List Nat := []
theorem row8329_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8329 row8329_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8329_large_left_left : List Nat := row8329_small ++ row8329_block_005
theorem row8329_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8329 row8329_large_left_left :=
  Budget.RowUpperBound.branch row8329_small_covers row8329_block_005_covers

def row8329_large_left_right_right : List Nat := row8329_block_007 ++ row8329_block_008
theorem row8329_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8329 row8329_large_left_right_right :=
  Budget.RowUpperBound.branch row8329_block_007_covers row8329_block_008_covers

def row8329_large_left_right : List Nat := row8329_block_006 ++ row8329_large_left_right_right
theorem row8329_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8329 row8329_large_left_right :=
  Budget.RowUpperBound.branch row8329_block_006_covers row8329_large_left_right_right_covers

def row8329_large_left : List Nat := row8329_large_left_left ++ row8329_large_left_right
theorem row8329_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8329 row8329_large_left :=
  Budget.RowUpperBound.branch row8329_large_left_left_covers row8329_large_left_right_covers

def row8329_large_right_left : List Nat := row8329_block_009 ++ row8329_block_010
theorem row8329_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8329 row8329_large_right_left :=
  Budget.RowUpperBound.branch row8329_block_009_covers row8329_block_010_covers

def row8329_large_right_right_right : List Nat := row8329_block_012 ++ row8329_block_013
theorem row8329_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8329 row8329_large_right_right_right :=
  Budget.RowUpperBound.branch row8329_block_012_covers row8329_block_013_covers

def row8329_large_right_right : List Nat := row8329_block_011 ++ row8329_large_right_right_right
theorem row8329_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8329 row8329_large_right_right :=
  Budget.RowUpperBound.branch row8329_block_011_covers row8329_large_right_right_right_covers

def row8329_large_right : List Nat := row8329_large_right_left ++ row8329_large_right_right
theorem row8329_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8329 row8329_large_right :=
  Budget.RowUpperBound.branch row8329_large_right_left_covers row8329_large_right_right_covers

def row8329_large : List Nat := row8329_large_left ++ row8329_large_right
theorem row8329_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8329 row8329_large :=
  Budget.RowUpperBound.branch row8329_large_left_covers row8329_large_right_covers

def row8329 : List Nat := row8329_large
theorem row8329_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8329 row8329 := row8329_large_covers

def row8353_small : List Nat := PrimePairUpper.smallRow8353
theorem row8353_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8353 row8353_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8353_covers

def row8353_block_005 : List Nat := [8419, 8713, 9829]
def plan8353_block_005 : List Budget.Choice := [(.firstFactor 11), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), (.firstFactor 17), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .second, .first, .second, (.firstFactor 31), (.firstFactor 13), (.secondFactor 37), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 29), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8353_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8353 row8353_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8353_block_005
  · decide
  · rfl

def row8353_block_006 : List Nat := [10639, 11047, 11251, 11317]
def plan8353_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8353_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8353 row8353_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8353_block_006
  · decide
  · rfl

def row8353_block_007 : List Nat := [13183, 13711, 13789]
def plan8353_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 11), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .second, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .second, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8353_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8353 row8353_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8353_block_007
  · decide
  · rfl

def row8353_block_008 : List Nat := [14551, 15619, 15643]
def plan8353_block_008 : List Budget.Choice := [(.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .possible, .first, .first, .possible, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 31), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8353_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8353 row8353_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8353_block_008
  · decide
  · rfl

def row8353_block_009 : List Nat := [17443]
def plan8353_block_009 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .possible, (.secondFactor 17), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8353_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8353 row8353_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8353_block_009
  · decide
  · rfl

def row8353_block_010 : List Nat := []
theorem row8353_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8353 row8353_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8353_block_011 : List Nat := []
theorem row8353_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8353 row8353_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8353_block_012 : List Nat := []
theorem row8353_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8353 row8353_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8353_block_013 : List Nat := []
theorem row8353_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8353 row8353_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8353_large_left_left : List Nat := row8353_small ++ row8353_block_005
theorem row8353_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8353 row8353_large_left_left :=
  Budget.RowUpperBound.branch row8353_small_covers row8353_block_005_covers

def row8353_large_left_right_right : List Nat := row8353_block_007 ++ row8353_block_008
theorem row8353_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8353 row8353_large_left_right_right :=
  Budget.RowUpperBound.branch row8353_block_007_covers row8353_block_008_covers

def row8353_large_left_right : List Nat := row8353_block_006 ++ row8353_large_left_right_right
theorem row8353_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8353 row8353_large_left_right :=
  Budget.RowUpperBound.branch row8353_block_006_covers row8353_large_left_right_right_covers

def row8353_large_left : List Nat := row8353_large_left_left ++ row8353_large_left_right
theorem row8353_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8353 row8353_large_left :=
  Budget.RowUpperBound.branch row8353_large_left_left_covers row8353_large_left_right_covers

def row8353_large_right_left : List Nat := row8353_block_009 ++ row8353_block_010
theorem row8353_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8353 row8353_large_right_left :=
  Budget.RowUpperBound.branch row8353_block_009_covers row8353_block_010_covers

def row8353_large_right_right_right : List Nat := row8353_block_012 ++ row8353_block_013
theorem row8353_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8353 row8353_large_right_right_right :=
  Budget.RowUpperBound.branch row8353_block_012_covers row8353_block_013_covers

def row8353_large_right_right : List Nat := row8353_block_011 ++ row8353_large_right_right_right
theorem row8353_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8353 row8353_large_right_right :=
  Budget.RowUpperBound.branch row8353_block_011_covers row8353_large_right_right_right_covers

def row8353_large_right : List Nat := row8353_large_right_left ++ row8353_large_right_right
theorem row8353_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8353 row8353_large_right :=
  Budget.RowUpperBound.branch row8353_large_right_left_covers row8353_large_right_right_covers

def row8353_large : List Nat := row8353_large_left ++ row8353_large_right
theorem row8353_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8353 row8353_large :=
  Budget.RowUpperBound.branch row8353_large_left_covers row8353_large_right_covers

def row8353 : List Nat := row8353_large
theorem row8353_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8353 row8353 := row8353_large_covers

end ProjectEulerSolutions.PrimePairLarge
