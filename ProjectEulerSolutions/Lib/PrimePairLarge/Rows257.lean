import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows080

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8221_small : List Nat := PrimePairUpper.smallRow8221
theorem row8221_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8221 row8221_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8221_covers

def row8221_block_005 : List Nat := [9127, 9433, 9649]
def plan8221_block_005 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .second, (.secondFactor 29), (.firstFactor 7), .possible, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, .possible, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .possible, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 31), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8221_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8221 row8221_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8221_block_005
  · decide
  · rfl

def row8221_block_006 : List Nat := [11443]
def plan8221_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8221_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8221 row8221_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8221_block_006
  · decide
  · rfl

def row8221_block_007 : List Nat := [12457, 13177, 13297, 14143, 14221]
def plan8221_block_007 : List Budget.Choice := [(.firstFactor 13), (.secondFactor 7), .first, .second, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 37), .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .possible, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .possible, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8221_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8221 row8221_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8221_block_007
  · decide
  · rfl

def row8221_block_008 : List Nat := []
def plan8221_block_008 : List Budget.Choice := [(.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), (.secondFactor 29), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .second, (.secondFactor 7), (.secondFactor 17), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8221_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8221 row8221_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8221_block_008
  · decide
  · rfl

def row8221_block_009 : List Nat := [16693, 16747, 17041, 17077, 17509]
def plan8221_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 31), .first, .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .possible, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .second, .first, .second, .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .possible, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8221_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8221 row8221_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8221_block_009
  · decide
  · rfl

def row8221_block_010 : List Nat := []
theorem row8221_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8221 row8221_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8221_block_011 : List Nat := []
theorem row8221_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8221 row8221_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8221_block_012 : List Nat := []
theorem row8221_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8221 row8221_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8221_block_013 : List Nat := []
theorem row8221_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8221 row8221_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8221_large_left_left : List Nat := row8221_small ++ row8221_block_005
theorem row8221_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8221 row8221_large_left_left :=
  Budget.RowUpperBound.branch row8221_small_covers row8221_block_005_covers

def row8221_large_left_right_right : List Nat := row8221_block_007 ++ row8221_block_008
theorem row8221_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8221 row8221_large_left_right_right :=
  Budget.RowUpperBound.branch row8221_block_007_covers row8221_block_008_covers

def row8221_large_left_right : List Nat := row8221_block_006 ++ row8221_large_left_right_right
theorem row8221_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8221 row8221_large_left_right :=
  Budget.RowUpperBound.branch row8221_block_006_covers row8221_large_left_right_right_covers

def row8221_large_left : List Nat := row8221_large_left_left ++ row8221_large_left_right
theorem row8221_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8221 row8221_large_left :=
  Budget.RowUpperBound.branch row8221_large_left_left_covers row8221_large_left_right_covers

def row8221_large_right_left : List Nat := row8221_block_009 ++ row8221_block_010
theorem row8221_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8221 row8221_large_right_left :=
  Budget.RowUpperBound.branch row8221_block_009_covers row8221_block_010_covers

def row8221_large_right_right_right : List Nat := row8221_block_012 ++ row8221_block_013
theorem row8221_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8221 row8221_large_right_right_right :=
  Budget.RowUpperBound.branch row8221_block_012_covers row8221_block_013_covers

def row8221_large_right_right : List Nat := row8221_block_011 ++ row8221_large_right_right_right
theorem row8221_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8221 row8221_large_right_right :=
  Budget.RowUpperBound.branch row8221_block_011_covers row8221_large_right_right_right_covers

def row8221_large_right : List Nat := row8221_large_right_left ++ row8221_large_right_right
theorem row8221_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8221 row8221_large_right :=
  Budget.RowUpperBound.branch row8221_large_right_left_covers row8221_large_right_right_covers

def row8221_large : List Nat := row8221_large_left ++ row8221_large_right
theorem row8221_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8221 row8221_large :=
  Budget.RowUpperBound.branch row8221_large_left_covers row8221_large_right_covers

def row8221 : List Nat := row8221_large
theorem row8221_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8221 row8221 := row8221_large_covers

def row8231_small : List Nat := PrimePairUpper.smallRow8231
theorem row8231_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8231 row8231_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8231_covers

def row8231_block_005 : List Nat := [9203, 9467, 9623, 9689]
def plan8231_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8231_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8231 row8231_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8231_block_005
  · decide
  · rfl

def row8231_block_006 : List Nat := [10487, 11087, 11897, 12041]
def plan8231_block_006 : List Budget.Choice := [(.secondFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .possible, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8231_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8231 row8231_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8231_block_006
  · decide
  · rfl

def row8231_block_007 : List Nat := [13037, 14249]
def plan8231_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8231_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8231 row8231_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8231_block_007
  · decide
  · rfl

def row8231_block_008 : List Nat := [16223]
def plan8231_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 13), .second, (.secondFactor 7), .first, .second, (.firstFactor 7), (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 31), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .possible, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8231_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8231 row8231_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8231_block_008
  · decide
  · rfl

def row8231_block_009 : List Nat := [17291]
def plan8231_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .second, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .second, .second, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .possible, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8231_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8231 row8231_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8231_block_009
  · decide
  · rfl

def row8231_block_010 : List Nat := []
theorem row8231_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8231 row8231_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8231_block_011 : List Nat := []
theorem row8231_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8231 row8231_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8231_block_012 : List Nat := []
theorem row8231_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8231 row8231_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8231_block_013 : List Nat := []
theorem row8231_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8231 row8231_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8231_large_left_left : List Nat := row8231_small ++ row8231_block_005
theorem row8231_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8231 row8231_large_left_left :=
  Budget.RowUpperBound.branch row8231_small_covers row8231_block_005_covers

def row8231_large_left_right_right : List Nat := row8231_block_007 ++ row8231_block_008
theorem row8231_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8231 row8231_large_left_right_right :=
  Budget.RowUpperBound.branch row8231_block_007_covers row8231_block_008_covers

def row8231_large_left_right : List Nat := row8231_block_006 ++ row8231_large_left_right_right
theorem row8231_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8231 row8231_large_left_right :=
  Budget.RowUpperBound.branch row8231_block_006_covers row8231_large_left_right_right_covers

def row8231_large_left : List Nat := row8231_large_left_left ++ row8231_large_left_right
theorem row8231_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8231 row8231_large_left :=
  Budget.RowUpperBound.branch row8231_large_left_left_covers row8231_large_left_right_covers

def row8231_large_right_left : List Nat := row8231_block_009 ++ row8231_block_010
theorem row8231_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8231 row8231_large_right_left :=
  Budget.RowUpperBound.branch row8231_block_009_covers row8231_block_010_covers

def row8231_large_right_right_right : List Nat := row8231_block_012 ++ row8231_block_013
theorem row8231_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8231 row8231_large_right_right_right :=
  Budget.RowUpperBound.branch row8231_block_012_covers row8231_block_013_covers

def row8231_large_right_right : List Nat := row8231_block_011 ++ row8231_large_right_right_right
theorem row8231_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8231 row8231_large_right_right :=
  Budget.RowUpperBound.branch row8231_block_011_covers row8231_large_right_right_right_covers

def row8231_large_right : List Nat := row8231_large_right_left ++ row8231_large_right_right
theorem row8231_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8231 row8231_large_right :=
  Budget.RowUpperBound.branch row8231_large_right_left_covers row8231_large_right_right_covers

def row8231_large : List Nat := row8231_large_left ++ row8231_large_right
theorem row8231_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8231 row8231_large :=
  Budget.RowUpperBound.branch row8231_large_left_covers row8231_large_right_covers

def row8231 : List Nat := row8231_large
theorem row8231_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8231 row8231 := row8231_large_covers

def row8233_small : List Nat := PrimePairUpper.smallRow8233
theorem row8233_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8233 row8233_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8233_covers

def row8233_block_005 : List Nat := [8563, 8677, 8863, 9067, 9199, 9769, 9949]
def plan8233_block_005 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 37), .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .possible, (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .possible, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .possible, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8233_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8233 row8233_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8233_block_005
  · decide
  · rfl

def row8233_block_006 : List Nat := [10597, 10651]
def plan8233_block_006 : List Budget.Choice := [.first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .possible, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8233_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8233 row8233_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8233_block_006
  · decide
  · rfl

def row8233_block_007 : List Nat := [13171, 13411]
def plan8233_block_007 : List Budget.Choice := [.second, .second, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 11), .first, .first, .second, (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 31), .first, .possible, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 17), .possible, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row8233_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8233 row8233_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8233_block_007
  · decide
  · rfl

def row8233_block_008 : List Nat := [14533, 15289, 15907]
def plan8233_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 29), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 11), (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), .second, .second, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8233_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8233 row8233_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8233_block_008
  · decide
  · rfl

def row8233_block_009 : List Nat := [17389, 17791]
def plan8233_block_009 : List Budget.Choice := [.second, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, .second, (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 37), (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8233_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8233 row8233_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8233_block_009
  · decide
  · rfl

def row8233_block_010 : List Nat := []
theorem row8233_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8233 row8233_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8233_block_011 : List Nat := []
theorem row8233_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8233 row8233_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8233_block_012 : List Nat := []
theorem row8233_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8233 row8233_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8233_block_013 : List Nat := []
theorem row8233_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8233 row8233_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8233_large_left_left : List Nat := row8233_small ++ row8233_block_005
theorem row8233_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8233 row8233_large_left_left :=
  Budget.RowUpperBound.branch row8233_small_covers row8233_block_005_covers

def row8233_large_left_right_right : List Nat := row8233_block_007 ++ row8233_block_008
theorem row8233_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8233 row8233_large_left_right_right :=
  Budget.RowUpperBound.branch row8233_block_007_covers row8233_block_008_covers

def row8233_large_left_right : List Nat := row8233_block_006 ++ row8233_large_left_right_right
theorem row8233_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8233 row8233_large_left_right :=
  Budget.RowUpperBound.branch row8233_block_006_covers row8233_large_left_right_right_covers

def row8233_large_left : List Nat := row8233_large_left_left ++ row8233_large_left_right
theorem row8233_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8233 row8233_large_left :=
  Budget.RowUpperBound.branch row8233_large_left_left_covers row8233_large_left_right_covers

def row8233_large_right_left : List Nat := row8233_block_009 ++ row8233_block_010
theorem row8233_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8233 row8233_large_right_left :=
  Budget.RowUpperBound.branch row8233_block_009_covers row8233_block_010_covers

def row8233_large_right_right_right : List Nat := row8233_block_012 ++ row8233_block_013
theorem row8233_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8233 row8233_large_right_right_right :=
  Budget.RowUpperBound.branch row8233_block_012_covers row8233_block_013_covers

def row8233_large_right_right : List Nat := row8233_block_011 ++ row8233_large_right_right_right
theorem row8233_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8233 row8233_large_right_right :=
  Budget.RowUpperBound.branch row8233_block_011_covers row8233_large_right_right_right_covers

def row8233_large_right : List Nat := row8233_large_right_left ++ row8233_large_right_right
theorem row8233_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8233 row8233_large_right :=
  Budget.RowUpperBound.branch row8233_large_right_left_covers row8233_large_right_right_covers

def row8233_large : List Nat := row8233_large_left ++ row8233_large_right
theorem row8233_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8233 row8233_large :=
  Budget.RowUpperBound.branch row8233_large_left_covers row8233_large_right_covers

def row8233 : List Nat := row8233_large
theorem row8233_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8233 row8233 := row8233_large_covers

def row8237_small : List Nat := PrimePairUpper.smallRow8237
theorem row8237_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8237 row8237_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8237_covers

def row8237_block_005 : List Nat := [9623, 9677, 9851, 10079, 10103, 10337]
def plan8237_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .second, .first, (.secondFactor 29), .first, (.secondFactor 37), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 23), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .possible, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .possible, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8237_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8237 row8237_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8237_block_005
  · decide
  · rfl

def row8237_block_006 : List Nat := [10739, 11177, 12119, 12269]
def plan8237_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 31), (.secondFactor 23), .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .possible, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8237_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8237 row8237_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8237_block_006
  · decide
  · rfl

def row8237_block_007 : List Nat := [12689, 12959, 13901]
def plan8237_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .second, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8237_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8237 row8237_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8237_block_007
  · decide
  · rfl

def row8237_block_008 : List Nat := [14561, 14897, 15791]
def plan8237_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 13), .possible, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .possible, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8237_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8237 row8237_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8237_block_008
  · decide
  · rfl

def row8237_block_009 : List Nat := [17333]
def plan8237_block_009 : List Budget.Choice := [.first, (.secondFactor 13), .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 31), (.secondFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .possible, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8237_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8237 row8237_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8237_block_009
  · decide
  · rfl

def row8237_block_010 : List Nat := []
theorem row8237_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8237 row8237_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8237_block_011 : List Nat := []
theorem row8237_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8237 row8237_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8237_block_012 : List Nat := []
theorem row8237_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8237 row8237_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8237_block_013 : List Nat := []
theorem row8237_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8237 row8237_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8237_large_left_left : List Nat := row8237_small ++ row8237_block_005
theorem row8237_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8237 row8237_large_left_left :=
  Budget.RowUpperBound.branch row8237_small_covers row8237_block_005_covers

def row8237_large_left_right_right : List Nat := row8237_block_007 ++ row8237_block_008
theorem row8237_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8237 row8237_large_left_right_right :=
  Budget.RowUpperBound.branch row8237_block_007_covers row8237_block_008_covers

def row8237_large_left_right : List Nat := row8237_block_006 ++ row8237_large_left_right_right
theorem row8237_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8237 row8237_large_left_right :=
  Budget.RowUpperBound.branch row8237_block_006_covers row8237_large_left_right_right_covers

def row8237_large_left : List Nat := row8237_large_left_left ++ row8237_large_left_right
theorem row8237_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8237 row8237_large_left :=
  Budget.RowUpperBound.branch row8237_large_left_left_covers row8237_large_left_right_covers

def row8237_large_right_left : List Nat := row8237_block_009 ++ row8237_block_010
theorem row8237_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8237 row8237_large_right_left :=
  Budget.RowUpperBound.branch row8237_block_009_covers row8237_block_010_covers

def row8237_large_right_right_right : List Nat := row8237_block_012 ++ row8237_block_013
theorem row8237_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8237 row8237_large_right_right_right :=
  Budget.RowUpperBound.branch row8237_block_012_covers row8237_block_013_covers

def row8237_large_right_right : List Nat := row8237_block_011 ++ row8237_large_right_right_right
theorem row8237_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8237 row8237_large_right_right :=
  Budget.RowUpperBound.branch row8237_block_011_covers row8237_large_right_right_right_covers

def row8237_large_right : List Nat := row8237_large_right_left ++ row8237_large_right_right
theorem row8237_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8237 row8237_large_right :=
  Budget.RowUpperBound.branch row8237_large_right_left_covers row8237_large_right_right_covers

def row8237_large : List Nat := row8237_large_left ++ row8237_large_right
theorem row8237_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8237 row8237_large :=
  Budget.RowUpperBound.branch row8237_large_left_covers row8237_large_right_covers

def row8237 : List Nat := row8237_large
theorem row8237_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8237 row8237 := row8237_large_covers

end ProjectEulerSolutions.PrimePairLarge
