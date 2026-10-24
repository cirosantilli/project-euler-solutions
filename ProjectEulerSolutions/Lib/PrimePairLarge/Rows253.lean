import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows078

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8081_small : List Nat := PrimePairUpper.smallRow8081
theorem row8081_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8081 row8081_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8081_covers

def row8081_block_005 : List Nat := [8537, 8753, 9161, 9419, 10253]
def plan8081_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 17), (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8081_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8081 row8081_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8081_block_005
  · decide
  · rfl

def row8081_block_006 : List Nat := [10559, 10847, 11027, 11867]
def plan8081_block_006 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 19), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8081_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8081 row8081_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8081_block_006
  · decide
  · rfl

def row8081_block_007 : List Nat := []
def plan8081_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 19), (.firstFactor 23), (.firstFactor 37), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 37), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 37), .first, (.secondFactor 13), .first, .second, .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .second, .first, .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8081_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8081 row8081_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8081_block_007
  · decide
  · rfl

def row8081_block_008 : List Nat := [15053, 15299, 15797, 16487]
def plan8081_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .possible, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 37), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .second, .first, (.secondFactor 7), .first, .first, .second, .possible, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8081_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8081 row8081_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8081_block_008
  · decide
  · rfl

def row8081_block_009 : List Nat := [16553]
def plan8081_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 23), .second, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 37), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8081_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8081 row8081_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8081_block_009
  · decide
  · rfl

def row8081_block_010 : List Nat := []
theorem row8081_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8081 row8081_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8081_block_011 : List Nat := []
theorem row8081_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8081 row8081_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8081_block_012 : List Nat := []
theorem row8081_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8081 row8081_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8081_block_013 : List Nat := []
theorem row8081_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8081 row8081_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8081_large_left_left : List Nat := row8081_small ++ row8081_block_005
theorem row8081_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8081 row8081_large_left_left :=
  Budget.RowUpperBound.branch row8081_small_covers row8081_block_005_covers

def row8081_large_left_right_right : List Nat := row8081_block_007 ++ row8081_block_008
theorem row8081_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8081 row8081_large_left_right_right :=
  Budget.RowUpperBound.branch row8081_block_007_covers row8081_block_008_covers

def row8081_large_left_right : List Nat := row8081_block_006 ++ row8081_large_left_right_right
theorem row8081_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8081 row8081_large_left_right :=
  Budget.RowUpperBound.branch row8081_block_006_covers row8081_large_left_right_right_covers

def row8081_large_left : List Nat := row8081_large_left_left ++ row8081_large_left_right
theorem row8081_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8081 row8081_large_left :=
  Budget.RowUpperBound.branch row8081_large_left_left_covers row8081_large_left_right_covers

def row8081_large_right_left : List Nat := row8081_block_009 ++ row8081_block_010
theorem row8081_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8081 row8081_large_right_left :=
  Budget.RowUpperBound.branch row8081_block_009_covers row8081_block_010_covers

def row8081_large_right_right_right : List Nat := row8081_block_012 ++ row8081_block_013
theorem row8081_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8081 row8081_large_right_right_right :=
  Budget.RowUpperBound.branch row8081_block_012_covers row8081_block_013_covers

def row8081_large_right_right : List Nat := row8081_block_011 ++ row8081_large_right_right_right
theorem row8081_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8081 row8081_large_right_right :=
  Budget.RowUpperBound.branch row8081_block_011_covers row8081_large_right_right_right_covers

def row8081_large_right : List Nat := row8081_large_right_left ++ row8081_large_right_right
theorem row8081_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8081 row8081_large_right :=
  Budget.RowUpperBound.branch row8081_large_right_left_covers row8081_large_right_right_covers

def row8081_large : List Nat := row8081_large_left ++ row8081_large_right
theorem row8081_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8081 row8081_large :=
  Budget.RowUpperBound.branch row8081_large_left_covers row8081_large_right_covers

def row8081 : List Nat := row8081_large
theorem row8081_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8081 row8081 := row8081_large_covers

def row8087_small : List Nat := PrimePairUpper.smallRow8087
theorem row8087_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8087 row8087_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8087_covers

def row8087_block_005 : List Nat := [9413, 9587]
def plan8087_block_005 : List Budget.Choice := [.first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .possible, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8087_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8087 row8087_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8087_block_005
  · decide
  · rfl

def row8087_block_006 : List Nat := [11807]
def plan8087_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 23), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8087_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8087 row8087_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8087_block_006
  · decide
  · rfl

def row8087_block_007 : List Nat := [13463, 13859, 13883, 13907, 14411]
def plan8087_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .second, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 19), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .possible, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .possible, (.secondFactor 13), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8087_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8087 row8087_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8087_block_007
  · decide
  · rfl

def row8087_block_008 : List Nat := [16193]
def plan8087_block_008 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8087_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8087 row8087_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8087_block_008
  · decide
  · rfl

def row8087_block_009 : List Nat := [17123]
def plan8087_block_009 : List Budget.Choice := [.first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .second, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8087_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8087 row8087_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8087_block_009
  · decide
  · rfl

def row8087_block_010 : List Nat := []
theorem row8087_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8087 row8087_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8087_block_011 : List Nat := []
theorem row8087_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8087 row8087_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8087_block_012 : List Nat := []
theorem row8087_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8087 row8087_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8087_block_013 : List Nat := []
theorem row8087_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8087 row8087_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8087_large_left_left : List Nat := row8087_small ++ row8087_block_005
theorem row8087_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8087 row8087_large_left_left :=
  Budget.RowUpperBound.branch row8087_small_covers row8087_block_005_covers

def row8087_large_left_right_right : List Nat := row8087_block_007 ++ row8087_block_008
theorem row8087_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8087 row8087_large_left_right_right :=
  Budget.RowUpperBound.branch row8087_block_007_covers row8087_block_008_covers

def row8087_large_left_right : List Nat := row8087_block_006 ++ row8087_large_left_right_right
theorem row8087_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8087 row8087_large_left_right :=
  Budget.RowUpperBound.branch row8087_block_006_covers row8087_large_left_right_right_covers

def row8087_large_left : List Nat := row8087_large_left_left ++ row8087_large_left_right
theorem row8087_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8087 row8087_large_left :=
  Budget.RowUpperBound.branch row8087_large_left_left_covers row8087_large_left_right_covers

def row8087_large_right_left : List Nat := row8087_block_009 ++ row8087_block_010
theorem row8087_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8087 row8087_large_right_left :=
  Budget.RowUpperBound.branch row8087_block_009_covers row8087_block_010_covers

def row8087_large_right_right_right : List Nat := row8087_block_012 ++ row8087_block_013
theorem row8087_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8087 row8087_large_right_right_right :=
  Budget.RowUpperBound.branch row8087_block_012_covers row8087_block_013_covers

def row8087_large_right_right : List Nat := row8087_block_011 ++ row8087_large_right_right_right
theorem row8087_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8087 row8087_large_right_right :=
  Budget.RowUpperBound.branch row8087_block_011_covers row8087_large_right_right_right_covers

def row8087_large_right : List Nat := row8087_large_right_left ++ row8087_large_right_right
theorem row8087_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8087 row8087_large_right :=
  Budget.RowUpperBound.branch row8087_large_right_left_covers row8087_large_right_right_covers

def row8087_large : List Nat := row8087_large_left ++ row8087_large_right
theorem row8087_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8087 row8087_large :=
  Budget.RowUpperBound.branch row8087_large_left_covers row8087_large_right_covers

def row8087 : List Nat := row8087_large
theorem row8087_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8087 row8087 := row8087_large_covers

def row8089_small : List Nat := PrimePairUpper.smallRow8089
theorem row8089_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8089 row8089_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8089_covers

def row8089_block_005 : List Nat := [8443, 8737, 8821, 8887, 9319, 9829, 10357]
def plan8089_block_005 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .second, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .second, .possible, .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .second, (.firstFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 37), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .second, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8089_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8089 row8089_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8089_block_005
  · decide
  · rfl

def row8089_block_006 : List Nat := [10903, 11113, 11197, 11503]
def plan8089_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 29), .possible, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8089_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8089 row8089_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8089_block_006
  · decide
  · rfl

def row8089_block_007 : List Nat := [12541, 12757, 12799, 14173]
def plan8089_block_007 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row8089_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8089 row8089_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8089_block_007
  · decide
  · rfl

def row8089_block_008 : List Nat := [14479, 14731, 14983, 15277, 15493, 15661, 16033, 16333]
def plan8089_block_008 : List Budget.Choice := [.possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .possible, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 31), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 31), .first, (.secondFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 19), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8089_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8089 row8089_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8089_block_008
  · decide
  · rfl

def row8089_block_009 : List Nat := [16651, 17713]
def plan8089_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .possible, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 17), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 19), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 11), .second, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8089_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8089 row8089_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8089_block_009
  · decide
  · rfl

def row8089_block_010 : List Nat := []
theorem row8089_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8089 row8089_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8089_block_011 : List Nat := []
theorem row8089_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8089 row8089_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8089_block_012 : List Nat := []
theorem row8089_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8089 row8089_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8089_block_013 : List Nat := []
theorem row8089_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8089 row8089_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8089_large_left_left : List Nat := row8089_small ++ row8089_block_005
theorem row8089_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8089 row8089_large_left_left :=
  Budget.RowUpperBound.branch row8089_small_covers row8089_block_005_covers

def row8089_large_left_right_right : List Nat := row8089_block_007 ++ row8089_block_008
theorem row8089_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8089 row8089_large_left_right_right :=
  Budget.RowUpperBound.branch row8089_block_007_covers row8089_block_008_covers

def row8089_large_left_right : List Nat := row8089_block_006 ++ row8089_large_left_right_right
theorem row8089_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8089 row8089_large_left_right :=
  Budget.RowUpperBound.branch row8089_block_006_covers row8089_large_left_right_right_covers

def row8089_large_left : List Nat := row8089_large_left_left ++ row8089_large_left_right
theorem row8089_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8089 row8089_large_left :=
  Budget.RowUpperBound.branch row8089_large_left_left_covers row8089_large_left_right_covers

def row8089_large_right_left : List Nat := row8089_block_009 ++ row8089_block_010
theorem row8089_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8089 row8089_large_right_left :=
  Budget.RowUpperBound.branch row8089_block_009_covers row8089_block_010_covers

def row8089_large_right_right_right : List Nat := row8089_block_012 ++ row8089_block_013
theorem row8089_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8089 row8089_large_right_right_right :=
  Budget.RowUpperBound.branch row8089_block_012_covers row8089_block_013_covers

def row8089_large_right_right : List Nat := row8089_block_011 ++ row8089_large_right_right_right
theorem row8089_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8089 row8089_large_right_right :=
  Budget.RowUpperBound.branch row8089_block_011_covers row8089_large_right_right_right_covers

def row8089_large_right : List Nat := row8089_large_right_left ++ row8089_large_right_right
theorem row8089_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8089 row8089_large_right :=
  Budget.RowUpperBound.branch row8089_large_right_left_covers row8089_large_right_right_covers

def row8089_large : List Nat := row8089_large_left ++ row8089_large_right
theorem row8089_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8089 row8089_large :=
  Budget.RowUpperBound.branch row8089_large_left_covers row8089_large_right_covers

def row8089 : List Nat := row8089_large
theorem row8089_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8089 row8089 := row8089_large_covers

def row8093_small : List Nat := PrimePairUpper.smallRow8093
theorem row8093_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8093 row8093_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8093_covers

def row8093_block_005 : List Nat := [8999, 9137]
def plan8093_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 37), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8093_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8093 row8093_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8093_block_005
  · decide
  · rfl

def row8093_block_006 : List Nat := [10457, 11483, 11801, 11813, 11939]
def plan8093_block_006 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .possible, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8093_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8093 row8093_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8093_block_006
  · decide
  · rfl

def row8093_block_007 : List Nat := [14423]
def plan8093_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 31), .second, .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8093_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8093 row8093_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8093_block_007
  · decide
  · rfl

def row8093_block_008 : List Nat := [14843, 14951, 15401, 15473, 15497]
def plan8093_block_008 : List Budget.Choice := [.first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), (.firstFactor 13), (.firstFactor 11), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .possible, (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .second, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .possible, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .possible, .first, .possible, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 31), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8093_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8093 row8093_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8093_block_008
  · decide
  · rfl

def row8093_block_009 : List Nat := []
def plan8093_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .second, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 37), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8093_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8093 row8093_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8093_block_009
  · decide
  · rfl

def row8093_block_010 : List Nat := []
theorem row8093_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8093 row8093_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8093_block_011 : List Nat := []
theorem row8093_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8093 row8093_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8093_block_012 : List Nat := []
theorem row8093_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8093 row8093_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8093_block_013 : List Nat := []
theorem row8093_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8093 row8093_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8093_large_left_left : List Nat := row8093_small ++ row8093_block_005
theorem row8093_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8093 row8093_large_left_left :=
  Budget.RowUpperBound.branch row8093_small_covers row8093_block_005_covers

def row8093_large_left_right_right : List Nat := row8093_block_007 ++ row8093_block_008
theorem row8093_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8093 row8093_large_left_right_right :=
  Budget.RowUpperBound.branch row8093_block_007_covers row8093_block_008_covers

def row8093_large_left_right : List Nat := row8093_block_006 ++ row8093_large_left_right_right
theorem row8093_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8093 row8093_large_left_right :=
  Budget.RowUpperBound.branch row8093_block_006_covers row8093_large_left_right_right_covers

def row8093_large_left : List Nat := row8093_large_left_left ++ row8093_large_left_right
theorem row8093_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8093 row8093_large_left :=
  Budget.RowUpperBound.branch row8093_large_left_left_covers row8093_large_left_right_covers

def row8093_large_right_left : List Nat := row8093_block_009 ++ row8093_block_010
theorem row8093_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8093 row8093_large_right_left :=
  Budget.RowUpperBound.branch row8093_block_009_covers row8093_block_010_covers

def row8093_large_right_right_right : List Nat := row8093_block_012 ++ row8093_block_013
theorem row8093_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8093 row8093_large_right_right_right :=
  Budget.RowUpperBound.branch row8093_block_012_covers row8093_block_013_covers

def row8093_large_right_right : List Nat := row8093_block_011 ++ row8093_large_right_right_right
theorem row8093_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8093 row8093_large_right_right :=
  Budget.RowUpperBound.branch row8093_block_011_covers row8093_large_right_right_right_covers

def row8093_large_right : List Nat := row8093_large_right_left ++ row8093_large_right_right
theorem row8093_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8093 row8093_large_right :=
  Budget.RowUpperBound.branch row8093_large_right_left_covers row8093_large_right_right_covers

def row8093_large : List Nat := row8093_large_left ++ row8093_large_right
theorem row8093_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8093 row8093_large :=
  Budget.RowUpperBound.branch row8093_large_left_covers row8093_large_right_covers

def row8093 : List Nat := row8093_large
theorem row8093_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8093 row8093 := row8093_large_covers

end ProjectEulerSolutions.PrimePairLarge
