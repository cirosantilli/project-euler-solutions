import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows082

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8363_small : List Nat := PrimePairUpper.smallRow8363
theorem row8363_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8363 row8363_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8363_covers

def row8363_block_005 : List Nat := [8543, 9029, 10067, 10253]
def plan8363_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .possible, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 37), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, .second, .first, .second, .first, (.firstFactor 23), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8363_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8363 row8363_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8363_block_005
  · decide
  · rfl

def row8363_block_006 : List Nat := [11423, 12011, 12143]
def plan8363_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 29), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 13), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .possible, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8363_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8363 row8363_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8363_block_006
  · decide
  · rfl

def row8363_block_007 : List Nat := [12437, 12941, 13217, 13337, 13781, 14153]
def plan8363_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .possible, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8363_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8363 row8363_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8363_block_007
  · decide
  · rfl

def row8363_block_008 : List Nat := [15101, 16361, 16493]
def plan8363_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 31), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .possible, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), .first, .second, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 13), .possible]
set_option maxRecDepth 2048 in
theorem row8363_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8363 row8363_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8363_block_008
  · decide
  · rfl

def row8363_block_009 : List Nat := [16763, 17627]
def plan8363_block_009 : List Budget.Choice := [.first, (.firstFactor 29), (.firstFactor 11), .second, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .possible, (.firstFactor 19), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .second, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 11), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8363_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8363 row8363_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8363_block_009
  · decide
  · rfl

def row8363_block_010 : List Nat := []
theorem row8363_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8363 row8363_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8363_block_011 : List Nat := []
theorem row8363_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8363 row8363_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8363_block_012 : List Nat := []
theorem row8363_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8363 row8363_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8363_block_013 : List Nat := []
theorem row8363_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8363 row8363_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8363_large_left_left : List Nat := row8363_small ++ row8363_block_005
theorem row8363_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8363 row8363_large_left_left :=
  Budget.RowUpperBound.branch row8363_small_covers row8363_block_005_covers

def row8363_large_left_right_right : List Nat := row8363_block_007 ++ row8363_block_008
theorem row8363_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8363 row8363_large_left_right_right :=
  Budget.RowUpperBound.branch row8363_block_007_covers row8363_block_008_covers

def row8363_large_left_right : List Nat := row8363_block_006 ++ row8363_large_left_right_right
theorem row8363_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8363 row8363_large_left_right :=
  Budget.RowUpperBound.branch row8363_block_006_covers row8363_large_left_right_right_covers

def row8363_large_left : List Nat := row8363_large_left_left ++ row8363_large_left_right
theorem row8363_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8363 row8363_large_left :=
  Budget.RowUpperBound.branch row8363_large_left_left_covers row8363_large_left_right_covers

def row8363_large_right_left : List Nat := row8363_block_009 ++ row8363_block_010
theorem row8363_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8363 row8363_large_right_left :=
  Budget.RowUpperBound.branch row8363_block_009_covers row8363_block_010_covers

def row8363_large_right_right_right : List Nat := row8363_block_012 ++ row8363_block_013
theorem row8363_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8363 row8363_large_right_right_right :=
  Budget.RowUpperBound.branch row8363_block_012_covers row8363_block_013_covers

def row8363_large_right_right : List Nat := row8363_block_011 ++ row8363_large_right_right_right
theorem row8363_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8363 row8363_large_right_right :=
  Budget.RowUpperBound.branch row8363_block_011_covers row8363_large_right_right_right_covers

def row8363_large_right : List Nat := row8363_large_right_left ++ row8363_large_right_right
theorem row8363_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8363 row8363_large_right :=
  Budget.RowUpperBound.branch row8363_large_right_left_covers row8363_large_right_right_covers

def row8363_large : List Nat := row8363_large_left ++ row8363_large_right
theorem row8363_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8363 row8363_large :=
  Budget.RowUpperBound.branch row8363_large_left_covers row8363_large_right_covers

def row8363 : List Nat := row8363_large
theorem row8363_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8363 row8363 := row8363_large_covers

def row8369_small : List Nat := PrimePairUpper.smallRow8369
theorem row8369_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8369 row8369_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8369_covers

def row8369_block_005 : List Nat := [8627, 8663, 9221, 10259]
def plan8369_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, (.secondFactor 23), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 23), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8369_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8369 row8369_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8369_block_005
  · decide
  · rfl

def row8369_block_006 : List Nat := [10631, 11027, 11897, 12119, 12281]
def plan8369_block_006 : List Budget.Choice := [(.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .possible, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, (.secondFactor 17), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .possible, (.secondFactor 13), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row8369_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8369 row8369_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8369_block_006
  · decide
  · rfl

def row8369_block_007 : List Nat := [13241, 13967]
def plan8369_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 19), (.firstFactor 7), .first, .second, (.secondFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 37), (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .second, .first, .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8369_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8369 row8369_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8369_block_007
  · decide
  · rfl

def row8369_block_008 : List Nat := [15461, 16193, 16349]
def plan8369_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 23), (.secondFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 23), (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 31), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8369_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8369 row8369_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8369_block_008
  · decide
  · rfl

def row8369_block_009 : List Nat := [17027, 17099, 17489]
def plan8369_block_009 : List Budget.Choice := [.first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8369_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8369 row8369_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8369_block_009
  · decide
  · rfl

def row8369_block_010 : List Nat := []
theorem row8369_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8369 row8369_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8369_block_011 : List Nat := []
theorem row8369_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8369 row8369_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8369_block_012 : List Nat := []
theorem row8369_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8369 row8369_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8369_block_013 : List Nat := []
theorem row8369_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8369 row8369_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8369_large_left_left : List Nat := row8369_small ++ row8369_block_005
theorem row8369_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8369 row8369_large_left_left :=
  Budget.RowUpperBound.branch row8369_small_covers row8369_block_005_covers

def row8369_large_left_right_right : List Nat := row8369_block_007 ++ row8369_block_008
theorem row8369_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8369 row8369_large_left_right_right :=
  Budget.RowUpperBound.branch row8369_block_007_covers row8369_block_008_covers

def row8369_large_left_right : List Nat := row8369_block_006 ++ row8369_large_left_right_right
theorem row8369_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8369 row8369_large_left_right :=
  Budget.RowUpperBound.branch row8369_block_006_covers row8369_large_left_right_right_covers

def row8369_large_left : List Nat := row8369_large_left_left ++ row8369_large_left_right
theorem row8369_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8369 row8369_large_left :=
  Budget.RowUpperBound.branch row8369_large_left_left_covers row8369_large_left_right_covers

def row8369_large_right_left : List Nat := row8369_block_009 ++ row8369_block_010
theorem row8369_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8369 row8369_large_right_left :=
  Budget.RowUpperBound.branch row8369_block_009_covers row8369_block_010_covers

def row8369_large_right_right_right : List Nat := row8369_block_012 ++ row8369_block_013
theorem row8369_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8369 row8369_large_right_right_right :=
  Budget.RowUpperBound.branch row8369_block_012_covers row8369_block_013_covers

def row8369_large_right_right : List Nat := row8369_block_011 ++ row8369_large_right_right_right
theorem row8369_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8369 row8369_large_right_right :=
  Budget.RowUpperBound.branch row8369_block_011_covers row8369_large_right_right_right_covers

def row8369_large_right : List Nat := row8369_large_right_left ++ row8369_large_right_right
theorem row8369_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8369 row8369_large_right :=
  Budget.RowUpperBound.branch row8369_large_right_left_covers row8369_large_right_right_covers

def row8369_large : List Nat := row8369_large_left ++ row8369_large_right
theorem row8369_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8369 row8369_large :=
  Budget.RowUpperBound.branch row8369_large_left_covers row8369_large_right_covers

def row8369 : List Nat := row8369_large
theorem row8369_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8369 row8369 := row8369_large_covers

def row8377_small : List Nat := PrimePairUpper.smallRow8377
theorem row8377_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8377 row8377_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8377_covers

def row8377_block_005 : List Nat := [8713, 9001, 9103, 9187, 10369]
def plan8377_block_005 : List Budget.Choice := [.first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 11), .possible, (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .second, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible]
set_option maxRecDepth 2048 in
theorem row8377_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8377 row8377_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8377_block_005
  · decide
  · rfl

def row8377_block_006 : List Nat := [11131, 11689]
def plan8377_block_006 : List Budget.Choice := [.first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8377_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8377 row8377_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8377_block_006
  · decide
  · rfl

def row8377_block_007 : List Nat := [12619, 13399, 14029]
def plan8377_block_007 : List Budget.Choice := [(.firstFactor 7), .second, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .second, .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 37), .second, .first, .second, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8377_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8377 row8377_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8377_block_007
  · decide
  · rfl

def row8377_block_008 : List Nat := [15349]
def plan8377_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, .second, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8377_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8377 row8377_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8377_block_008
  · decide
  · rfl

def row8377_block_009 : List Nat := [17191]
def plan8377_block_009 : List Budget.Choice := [(.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 17), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8377_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8377 row8377_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8377_block_009
  · decide
  · rfl

def row8377_block_010 : List Nat := []
theorem row8377_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8377 row8377_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8377_block_011 : List Nat := []
theorem row8377_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8377 row8377_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8377_block_012 : List Nat := []
theorem row8377_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8377 row8377_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8377_block_013 : List Nat := []
theorem row8377_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8377 row8377_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8377_large_left_left : List Nat := row8377_small ++ row8377_block_005
theorem row8377_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8377 row8377_large_left_left :=
  Budget.RowUpperBound.branch row8377_small_covers row8377_block_005_covers

def row8377_large_left_right_right : List Nat := row8377_block_007 ++ row8377_block_008
theorem row8377_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8377 row8377_large_left_right_right :=
  Budget.RowUpperBound.branch row8377_block_007_covers row8377_block_008_covers

def row8377_large_left_right : List Nat := row8377_block_006 ++ row8377_large_left_right_right
theorem row8377_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8377 row8377_large_left_right :=
  Budget.RowUpperBound.branch row8377_block_006_covers row8377_large_left_right_right_covers

def row8377_large_left : List Nat := row8377_large_left_left ++ row8377_large_left_right
theorem row8377_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8377 row8377_large_left :=
  Budget.RowUpperBound.branch row8377_large_left_left_covers row8377_large_left_right_covers

def row8377_large_right_left : List Nat := row8377_block_009 ++ row8377_block_010
theorem row8377_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8377 row8377_large_right_left :=
  Budget.RowUpperBound.branch row8377_block_009_covers row8377_block_010_covers

def row8377_large_right_right_right : List Nat := row8377_block_012 ++ row8377_block_013
theorem row8377_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8377 row8377_large_right_right_right :=
  Budget.RowUpperBound.branch row8377_block_012_covers row8377_block_013_covers

def row8377_large_right_right : List Nat := row8377_block_011 ++ row8377_large_right_right_right
theorem row8377_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8377 row8377_large_right_right :=
  Budget.RowUpperBound.branch row8377_block_011_covers row8377_large_right_right_right_covers

def row8377_large_right : List Nat := row8377_large_right_left ++ row8377_large_right_right
theorem row8377_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8377 row8377_large_right :=
  Budget.RowUpperBound.branch row8377_large_right_left_covers row8377_large_right_right_covers

def row8377_large : List Nat := row8377_large_left ++ row8377_large_right
theorem row8377_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8377 row8377_large :=
  Budget.RowUpperBound.branch row8377_large_left_covers row8377_large_right_covers

def row8377 : List Nat := row8377_large
theorem row8377_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8377 row8377 := row8377_large_covers

def row8387_small : List Nat := PrimePairUpper.smallRow8387
theorem row8387_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8387 row8387_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8387_covers

def row8387_block_005 : List Nat := [8849, 9491]
def plan8387_block_005 : List Budget.Choice := [.first, .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .second, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 17), .possible, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 29), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8387_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8387 row8387_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8387_block_005
  · decide
  · rfl

def row8387_block_006 : List Nat := [10937, 11579, 11717]
def plan8387_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8387_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8387 row8387_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8387_block_006
  · decide
  · rfl

def row8387_block_007 : List Nat := [12473, 13121]
def plan8387_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 23), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row8387_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8387 row8387_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8387_block_007
  · decide
  · rfl

def row8387_block_008 : List Nat := [16127, 16139]
def plan8387_block_008 : List Budget.Choice := [.first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 17), (.secondFactor 23), .first, .possible, .possible, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8387_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8387 row8387_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8387_block_008
  · decide
  · rfl

def row8387_block_009 : List Nat := [16883, 17027]
def plan8387_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .possible, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8387_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8387 row8387_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8387_block_009
  · decide
  · rfl

def row8387_block_010 : List Nat := []
theorem row8387_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8387 row8387_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8387_block_011 : List Nat := []
theorem row8387_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8387 row8387_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8387_block_012 : List Nat := []
theorem row8387_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8387 row8387_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8387_block_013 : List Nat := []
theorem row8387_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8387 row8387_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8387_large_left_left : List Nat := row8387_small ++ row8387_block_005
theorem row8387_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8387 row8387_large_left_left :=
  Budget.RowUpperBound.branch row8387_small_covers row8387_block_005_covers

def row8387_large_left_right_right : List Nat := row8387_block_007 ++ row8387_block_008
theorem row8387_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8387 row8387_large_left_right_right :=
  Budget.RowUpperBound.branch row8387_block_007_covers row8387_block_008_covers

def row8387_large_left_right : List Nat := row8387_block_006 ++ row8387_large_left_right_right
theorem row8387_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8387 row8387_large_left_right :=
  Budget.RowUpperBound.branch row8387_block_006_covers row8387_large_left_right_right_covers

def row8387_large_left : List Nat := row8387_large_left_left ++ row8387_large_left_right
theorem row8387_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8387 row8387_large_left :=
  Budget.RowUpperBound.branch row8387_large_left_left_covers row8387_large_left_right_covers

def row8387_large_right_left : List Nat := row8387_block_009 ++ row8387_block_010
theorem row8387_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8387 row8387_large_right_left :=
  Budget.RowUpperBound.branch row8387_block_009_covers row8387_block_010_covers

def row8387_large_right_right_right : List Nat := row8387_block_012 ++ row8387_block_013
theorem row8387_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8387 row8387_large_right_right_right :=
  Budget.RowUpperBound.branch row8387_block_012_covers row8387_block_013_covers

def row8387_large_right_right : List Nat := row8387_block_011 ++ row8387_large_right_right_right
theorem row8387_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8387 row8387_large_right_right :=
  Budget.RowUpperBound.branch row8387_block_011_covers row8387_large_right_right_right_covers

def row8387_large_right : List Nat := row8387_large_right_left ++ row8387_large_right_right
theorem row8387_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8387 row8387_large_right :=
  Budget.RowUpperBound.branch row8387_large_right_left_covers row8387_large_right_right_covers

def row8387_large : List Nat := row8387_large_left ++ row8387_large_right
theorem row8387_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8387 row8387_large :=
  Budget.RowUpperBound.branch row8387_large_left_covers row8387_large_right_covers

def row8387 : List Nat := row8387_large
theorem row8387_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8387 row8387 := row8387_large_covers

end ProjectEulerSolutions.PrimePairLarge
