import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows081

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8243_small : List Nat := PrimePairUpper.smallRow8243
theorem row8243_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8243 row8243_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8243_covers

def row8243_block_005 : List Nat := [8963, 9209, 9473, 9587, 9629]
def plan8243_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 17), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .second, .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .possible, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .possible, (.secondFactor 13), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 37), .second, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8243_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8243 row8243_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8243_block_005
  · decide
  · rfl

def row8243_block_006 : List Nat := [11177, 11321]
def plan8243_block_006 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .second, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 23), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 17), .second, .first, .first, .first, .second, .first, .second, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row8243_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8243 row8243_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8243_block_006
  · decide
  · rfl

def row8243_block_007 : List Nat := [12941, 13751, 13907, 14159, 14423]
def plan8243_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 37), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .possible, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 31), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .possible, .first, .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8243_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8243 row8243_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8243_block_007
  · decide
  · rfl

def row8243_block_008 : List Nat := [14537, 15083, 15299]
def plan8243_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 31), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 31), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8243_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8243 row8243_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8243_block_008
  · decide
  · rfl

def row8243_block_009 : List Nat := [17657]
def plan8243_block_009 : List Budget.Choice := [.first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8243_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8243 row8243_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8243_block_009
  · decide
  · rfl

def row8243_block_010 : List Nat := []
theorem row8243_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8243 row8243_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8243_block_011 : List Nat := []
theorem row8243_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8243 row8243_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8243_block_012 : List Nat := []
theorem row8243_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8243 row8243_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8243_block_013 : List Nat := []
theorem row8243_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8243 row8243_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8243_large_left_left : List Nat := row8243_small ++ row8243_block_005
theorem row8243_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8243 row8243_large_left_left :=
  Budget.RowUpperBound.branch row8243_small_covers row8243_block_005_covers

def row8243_large_left_right_right : List Nat := row8243_block_007 ++ row8243_block_008
theorem row8243_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8243 row8243_large_left_right_right :=
  Budget.RowUpperBound.branch row8243_block_007_covers row8243_block_008_covers

def row8243_large_left_right : List Nat := row8243_block_006 ++ row8243_large_left_right_right
theorem row8243_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8243 row8243_large_left_right :=
  Budget.RowUpperBound.branch row8243_block_006_covers row8243_large_left_right_right_covers

def row8243_large_left : List Nat := row8243_large_left_left ++ row8243_large_left_right
theorem row8243_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8243 row8243_large_left :=
  Budget.RowUpperBound.branch row8243_large_left_left_covers row8243_large_left_right_covers

def row8243_large_right_left : List Nat := row8243_block_009 ++ row8243_block_010
theorem row8243_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8243 row8243_large_right_left :=
  Budget.RowUpperBound.branch row8243_block_009_covers row8243_block_010_covers

def row8243_large_right_right_right : List Nat := row8243_block_012 ++ row8243_block_013
theorem row8243_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8243 row8243_large_right_right_right :=
  Budget.RowUpperBound.branch row8243_block_012_covers row8243_block_013_covers

def row8243_large_right_right : List Nat := row8243_block_011 ++ row8243_large_right_right_right
theorem row8243_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8243 row8243_large_right_right :=
  Budget.RowUpperBound.branch row8243_block_011_covers row8243_large_right_right_right_covers

def row8243_large_right : List Nat := row8243_large_right_left ++ row8243_large_right_right
theorem row8243_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8243 row8243_large_right :=
  Budget.RowUpperBound.branch row8243_large_right_left_covers row8243_large_right_right_covers

def row8243_large : List Nat := row8243_large_left ++ row8243_large_right
theorem row8243_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8243 row8243_large :=
  Budget.RowUpperBound.branch row8243_large_left_covers row8243_large_right_covers

def row8243 : List Nat := row8243_large
theorem row8243_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8243 row8243 := row8243_large_covers

def row8263_small : List Nat := PrimePairUpper.smallRow8263
theorem row8263_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8263 row8263_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8263_covers

def row8263_block_005 : List Nat := [9343, 9433, 9511, 9547, 9967]
def plan8263_block_005 : List Budget.Choice := [.first, (.secondFactor 17), .first, .first, .second, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 29), (.firstFactor 19), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 17), .second, (.firstFactor 7), .second, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 29), .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8263_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8263 row8263_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8263_block_005
  · decide
  · rfl

def row8263_block_006 : List Nat := []
def plan8263_block_006 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 31), (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8263_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8263 row8263_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8263_block_006
  · decide
  · rfl

def row8263_block_007 : List Nat := [13003, 13633, 13681, 13729, 14143, 14173, 14341]
def plan8263_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .second, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 29), .first, .first, .possible, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 11), .possible, .first, (.secondFactor 13), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .possible, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .possible, (.secondFactor 19), .first, .first, .possible, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8263_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8263 row8263_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8263_block_007
  · decide
  · rfl

def row8263_block_008 : List Nat := [14737, 15313]
def plan8263_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 13), .possible, (.firstFactor 23), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .second, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 11), .second, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8263_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8263 row8263_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8263_block_008
  · decide
  · rfl

def row8263_block_009 : List Nat := [16567, 16903, 17203, 17377, 17509, 17659]
def plan8263_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .second, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), (.firstFactor 7), (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .possible, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8263_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8263 row8263_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8263_block_009
  · decide
  · rfl

def row8263_block_010 : List Nat := []
theorem row8263_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8263 row8263_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8263_block_011 : List Nat := []
theorem row8263_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8263 row8263_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8263_block_012 : List Nat := []
theorem row8263_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8263 row8263_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8263_block_013 : List Nat := []
theorem row8263_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8263 row8263_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8263_large_left_left : List Nat := row8263_small ++ row8263_block_005
theorem row8263_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8263 row8263_large_left_left :=
  Budget.RowUpperBound.branch row8263_small_covers row8263_block_005_covers

def row8263_large_left_right_right : List Nat := row8263_block_007 ++ row8263_block_008
theorem row8263_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8263 row8263_large_left_right_right :=
  Budget.RowUpperBound.branch row8263_block_007_covers row8263_block_008_covers

def row8263_large_left_right : List Nat := row8263_block_006 ++ row8263_large_left_right_right
theorem row8263_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8263 row8263_large_left_right :=
  Budget.RowUpperBound.branch row8263_block_006_covers row8263_large_left_right_right_covers

def row8263_large_left : List Nat := row8263_large_left_left ++ row8263_large_left_right
theorem row8263_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8263 row8263_large_left :=
  Budget.RowUpperBound.branch row8263_large_left_left_covers row8263_large_left_right_covers

def row8263_large_right_left : List Nat := row8263_block_009 ++ row8263_block_010
theorem row8263_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8263 row8263_large_right_left :=
  Budget.RowUpperBound.branch row8263_block_009_covers row8263_block_010_covers

def row8263_large_right_right_right : List Nat := row8263_block_012 ++ row8263_block_013
theorem row8263_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8263 row8263_large_right_right_right :=
  Budget.RowUpperBound.branch row8263_block_012_covers row8263_block_013_covers

def row8263_large_right_right : List Nat := row8263_block_011 ++ row8263_large_right_right_right
theorem row8263_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8263 row8263_large_right_right :=
  Budget.RowUpperBound.branch row8263_block_011_covers row8263_large_right_right_right_covers

def row8263_large_right : List Nat := row8263_large_right_left ++ row8263_large_right_right
theorem row8263_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8263 row8263_large_right :=
  Budget.RowUpperBound.branch row8263_large_right_left_covers row8263_large_right_right_covers

def row8263_large : List Nat := row8263_large_left ++ row8263_large_right
theorem row8263_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8263 row8263_large :=
  Budget.RowUpperBound.branch row8263_large_left_covers row8263_large_right_covers

def row8263 : List Nat := row8263_large
theorem row8263_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8263 row8263 := row8263_large_covers

def row8269_small : List Nat := PrimePairUpper.smallRow8269
theorem row8269_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8269 row8269_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8269_covers

def row8269_block_005 : List Nat := [8527, 8713, 8821, 9319, 9787]
def plan8269_block_005 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 29), .possible, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 29), .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8269_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8269 row8269_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8269_block_005
  · decide
  · rfl

def row8269_block_006 : List Nat := [11383, 11941, 12049]
def plan8269_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), .second, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row8269_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8269 row8269_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8269_block_006
  · decide
  · rfl

def row8269_block_007 : List Nat := [12577, 12637, 12781, 13873, 13903, 14107]
def plan8269_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .possible, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 11), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 11), .first, .possible, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row8269_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8269 row8269_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8269_block_007
  · decide
  · rfl

def row8269_block_008 : List Nat := [14779, 15121, 15217, 16231, 16477]
def plan8269_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .second, .first, .first, .possible, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .possible, .first, .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8269_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8269 row8269_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8269_block_008
  · decide
  · rfl

def row8269_block_009 : List Nat := [17239, 17431, 17737]
def plan8269_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.secondFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 11), (.firstFactor 19), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8269_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8269 row8269_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8269_block_009
  · decide
  · rfl

def row8269_block_010 : List Nat := []
theorem row8269_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8269 row8269_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8269_block_011 : List Nat := []
theorem row8269_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8269 row8269_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8269_block_012 : List Nat := []
theorem row8269_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8269 row8269_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8269_block_013 : List Nat := []
theorem row8269_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8269 row8269_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8269_large_left_left : List Nat := row8269_small ++ row8269_block_005
theorem row8269_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8269 row8269_large_left_left :=
  Budget.RowUpperBound.branch row8269_small_covers row8269_block_005_covers

def row8269_large_left_right_right : List Nat := row8269_block_007 ++ row8269_block_008
theorem row8269_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8269 row8269_large_left_right_right :=
  Budget.RowUpperBound.branch row8269_block_007_covers row8269_block_008_covers

def row8269_large_left_right : List Nat := row8269_block_006 ++ row8269_large_left_right_right
theorem row8269_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8269 row8269_large_left_right :=
  Budget.RowUpperBound.branch row8269_block_006_covers row8269_large_left_right_right_covers

def row8269_large_left : List Nat := row8269_large_left_left ++ row8269_large_left_right
theorem row8269_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8269 row8269_large_left :=
  Budget.RowUpperBound.branch row8269_large_left_left_covers row8269_large_left_right_covers

def row8269_large_right_left : List Nat := row8269_block_009 ++ row8269_block_010
theorem row8269_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8269 row8269_large_right_left :=
  Budget.RowUpperBound.branch row8269_block_009_covers row8269_block_010_covers

def row8269_large_right_right_right : List Nat := row8269_block_012 ++ row8269_block_013
theorem row8269_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8269 row8269_large_right_right_right :=
  Budget.RowUpperBound.branch row8269_block_012_covers row8269_block_013_covers

def row8269_large_right_right : List Nat := row8269_block_011 ++ row8269_large_right_right_right
theorem row8269_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8269 row8269_large_right_right :=
  Budget.RowUpperBound.branch row8269_block_011_covers row8269_large_right_right_right_covers

def row8269_large_right : List Nat := row8269_large_right_left ++ row8269_large_right_right
theorem row8269_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8269 row8269_large_right :=
  Budget.RowUpperBound.branch row8269_large_right_left_covers row8269_large_right_right_covers

def row8269_large : List Nat := row8269_large_left ++ row8269_large_right
theorem row8269_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8269 row8269_large :=
  Budget.RowUpperBound.branch row8269_large_left_covers row8269_large_right_covers

def row8269 : List Nat := row8269_large
theorem row8269_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8269 row8269 := row8269_large_covers

def row8273_small : List Nat := PrimePairUpper.smallRow8273
theorem row8273_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8273 row8273_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8273_covers

def row8273_block_005 : List Nat := [8543, 8681, 9227, 10007, 10079]
def plan8273_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .possible, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.secondFactor 17), .second, (.secondFactor 13), .first, .first, (.firstFactor 23), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, .first]
set_option maxRecDepth 2048 in
theorem row8273_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8273 row8273_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8273_block_005
  · decide
  · rfl

def row8273_block_006 : List Nat := [11423, 11717, 11831, 12113, 12347]
def plan8273_block_006 : List Budget.Choice := [(.firstFactor 23), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .second, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 7), (.firstFactor 19), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 37), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .possible, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8273_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8273 row8273_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8273_block_006
  · decide
  · rfl

def row8273_block_007 : List Nat := [13217]
def plan8273_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .second, (.secondFactor 19), (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .second, .possible, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 31), .first, .second, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 29), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8273_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8273 row8273_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8273_block_007
  · decide
  · rfl

def row8273_block_008 : List Nat := [14627, 14759, 14771, 15131, 15767]
def plan8273_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .possible, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 29), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 17), .possible, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8273_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8273 row8273_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8273_block_008
  · decide
  · rfl

def row8273_block_009 : List Nat := [17657]
def plan8273_block_009 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.secondFactor 31), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, .second, .first, .first, .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 19), .possible, .first, (.firstFactor 37), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8273_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8273 row8273_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8273_block_009
  · decide
  · rfl

def row8273_block_010 : List Nat := []
theorem row8273_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8273 row8273_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8273_block_011 : List Nat := []
theorem row8273_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8273 row8273_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8273_block_012 : List Nat := []
theorem row8273_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8273 row8273_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8273_block_013 : List Nat := []
theorem row8273_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8273 row8273_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8273_large_left_left : List Nat := row8273_small ++ row8273_block_005
theorem row8273_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8273 row8273_large_left_left :=
  Budget.RowUpperBound.branch row8273_small_covers row8273_block_005_covers

def row8273_large_left_right_right : List Nat := row8273_block_007 ++ row8273_block_008
theorem row8273_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8273 row8273_large_left_right_right :=
  Budget.RowUpperBound.branch row8273_block_007_covers row8273_block_008_covers

def row8273_large_left_right : List Nat := row8273_block_006 ++ row8273_large_left_right_right
theorem row8273_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8273 row8273_large_left_right :=
  Budget.RowUpperBound.branch row8273_block_006_covers row8273_large_left_right_right_covers

def row8273_large_left : List Nat := row8273_large_left_left ++ row8273_large_left_right
theorem row8273_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8273 row8273_large_left :=
  Budget.RowUpperBound.branch row8273_large_left_left_covers row8273_large_left_right_covers

def row8273_large_right_left : List Nat := row8273_block_009 ++ row8273_block_010
theorem row8273_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8273 row8273_large_right_left :=
  Budget.RowUpperBound.branch row8273_block_009_covers row8273_block_010_covers

def row8273_large_right_right_right : List Nat := row8273_block_012 ++ row8273_block_013
theorem row8273_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8273 row8273_large_right_right_right :=
  Budget.RowUpperBound.branch row8273_block_012_covers row8273_block_013_covers

def row8273_large_right_right : List Nat := row8273_block_011 ++ row8273_large_right_right_right
theorem row8273_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8273 row8273_large_right_right :=
  Budget.RowUpperBound.branch row8273_block_011_covers row8273_large_right_right_right_covers

def row8273_large_right : List Nat := row8273_large_right_left ++ row8273_large_right_right
theorem row8273_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8273 row8273_large_right :=
  Budget.RowUpperBound.branch row8273_large_right_left_covers row8273_large_right_right_covers

def row8273_large : List Nat := row8273_large_left ++ row8273_large_right
theorem row8273_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8273 row8273_large :=
  Budget.RowUpperBound.branch row8273_large_left_covers row8273_large_right_covers

def row8273 : List Nat := row8273_large
theorem row8273_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8273 row8273 := row8273_large_covers

end ProjectEulerSolutions.PrimePairLarge
