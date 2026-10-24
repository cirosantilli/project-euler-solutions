import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows079

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8101_small : List Nat := PrimePairUpper.smallRow8101
theorem row8101_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8101 row8101_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8101_covers

def row8101_block_005 : List Nat := [8467, 9241, 9349, 9643, 9721]
def plan8101_block_005 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 31), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .second, .first, .first, .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .second, .second, (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .second, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 37), .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8101_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8101 row8101_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8101_block_005
  · decide
  · rfl

def row8101_block_006 : List Nat := [10501, 11173, 11503, 11719]
def plan8101_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8101_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8101 row8101_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8101_block_006
  · decide
  · rfl

def row8101_block_007 : List Nat := [12451, 12703, 12889, 13309, 13441, 13537, 13873, 14407]
def plan8101_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 13), .first, .possible, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 31), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 23), .possible, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .second, (.secondFactor 7), (.firstFactor 11), .possible, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8101_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8101 row8101_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8101_block_007
  · decide
  · rfl

def row8101_block_008 : List Nat := [14827, 15391, 15583, 16477]
def plan8101_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .second, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .second, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .possible, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8101_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8101 row8101_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8101_block_008
  · decide
  · rfl

def row8101_block_009 : List Nat := []
def plan8101_block_009 : List Budget.Choice := [(.secondFactor 31), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 31), (.secondFactor 13), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 29), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8101_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8101 row8101_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8101_block_009
  · decide
  · rfl

def row8101_block_010 : List Nat := []
theorem row8101_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8101 row8101_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8101_block_011 : List Nat := []
theorem row8101_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8101 row8101_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8101_block_012 : List Nat := []
theorem row8101_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8101 row8101_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8101_block_013 : List Nat := []
theorem row8101_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8101 row8101_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8101_large_left_left : List Nat := row8101_small ++ row8101_block_005
theorem row8101_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8101 row8101_large_left_left :=
  Budget.RowUpperBound.branch row8101_small_covers row8101_block_005_covers

def row8101_large_left_right_right : List Nat := row8101_block_007 ++ row8101_block_008
theorem row8101_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8101 row8101_large_left_right_right :=
  Budget.RowUpperBound.branch row8101_block_007_covers row8101_block_008_covers

def row8101_large_left_right : List Nat := row8101_block_006 ++ row8101_large_left_right_right
theorem row8101_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8101 row8101_large_left_right :=
  Budget.RowUpperBound.branch row8101_block_006_covers row8101_large_left_right_right_covers

def row8101_large_left : List Nat := row8101_large_left_left ++ row8101_large_left_right
theorem row8101_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8101 row8101_large_left :=
  Budget.RowUpperBound.branch row8101_large_left_left_covers row8101_large_left_right_covers

def row8101_large_right_left : List Nat := row8101_block_009 ++ row8101_block_010
theorem row8101_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8101 row8101_large_right_left :=
  Budget.RowUpperBound.branch row8101_block_009_covers row8101_block_010_covers

def row8101_large_right_right_right : List Nat := row8101_block_012 ++ row8101_block_013
theorem row8101_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8101 row8101_large_right_right_right :=
  Budget.RowUpperBound.branch row8101_block_012_covers row8101_block_013_covers

def row8101_large_right_right : List Nat := row8101_block_011 ++ row8101_large_right_right_right
theorem row8101_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8101 row8101_large_right_right :=
  Budget.RowUpperBound.branch row8101_block_011_covers row8101_large_right_right_right_covers

def row8101_large_right : List Nat := row8101_large_right_left ++ row8101_large_right_right
theorem row8101_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8101 row8101_large_right :=
  Budget.RowUpperBound.branch row8101_large_right_left_covers row8101_large_right_right_covers

def row8101_large : List Nat := row8101_large_left ++ row8101_large_right
theorem row8101_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8101 row8101_large :=
  Budget.RowUpperBound.branch row8101_large_left_covers row8101_large_right_covers

def row8101 : List Nat := row8101_large
theorem row8101_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8101 row8101 := row8101_large_covers

def row8111_small : List Nat := PrimePairUpper.smallRow8111
theorem row8111_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8111 row8111_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8111_covers

def row8111_block_005 : List Nat := [8573, 8963, 9221, 9743, 10151]
def plan8111_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .possible, .second, .first, (.firstFactor 13), .first, .first, .second, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .possible, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .possible, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 7), .first, .possible, .first, .second, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 37), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8111_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8111 row8111_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8111_block_005
  · decide
  · rfl

def row8111_block_006 : List Nat := []
def plan8111_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), .second, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 31), (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .second, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 31), (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8111_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8111 row8111_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8111_block_006
  · decide
  · rfl

def row8111_block_007 : List Nat := [12941, 13103]
def plan8111_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 13), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 31), .possible, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .possible, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .second, (.firstFactor 17), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .second, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, .first, .first, .second, .second, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8111_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8111 row8111_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8111_block_007
  · decide
  · rfl

def row8111_block_008 : List Nat := [14639, 15269, 15497, 16253]
def plan8111_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .second, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8111_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8111 row8111_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8111_block_008
  · decide
  · rfl

def row8111_block_009 : List Nat := [16547, 17597, 17789]
def plan8111_block_009 : List Budget.Choice := [.first, (.firstFactor 17), .possible, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), (.firstFactor 31), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .possible, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 17), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8111_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8111 row8111_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8111_block_009
  · decide
  · rfl

def row8111_block_010 : List Nat := []
theorem row8111_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8111 row8111_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8111_block_011 : List Nat := []
theorem row8111_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8111 row8111_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8111_block_012 : List Nat := []
theorem row8111_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8111 row8111_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8111_block_013 : List Nat := []
theorem row8111_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8111 row8111_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8111_large_left_left : List Nat := row8111_small ++ row8111_block_005
theorem row8111_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8111 row8111_large_left_left :=
  Budget.RowUpperBound.branch row8111_small_covers row8111_block_005_covers

def row8111_large_left_right_right : List Nat := row8111_block_007 ++ row8111_block_008
theorem row8111_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8111 row8111_large_left_right_right :=
  Budget.RowUpperBound.branch row8111_block_007_covers row8111_block_008_covers

def row8111_large_left_right : List Nat := row8111_block_006 ++ row8111_large_left_right_right
theorem row8111_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8111 row8111_large_left_right :=
  Budget.RowUpperBound.branch row8111_block_006_covers row8111_large_left_right_right_covers

def row8111_large_left : List Nat := row8111_large_left_left ++ row8111_large_left_right
theorem row8111_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8111 row8111_large_left :=
  Budget.RowUpperBound.branch row8111_large_left_left_covers row8111_large_left_right_covers

def row8111_large_right_left : List Nat := row8111_block_009 ++ row8111_block_010
theorem row8111_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8111 row8111_large_right_left :=
  Budget.RowUpperBound.branch row8111_block_009_covers row8111_block_010_covers

def row8111_large_right_right_right : List Nat := row8111_block_012 ++ row8111_block_013
theorem row8111_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8111 row8111_large_right_right_right :=
  Budget.RowUpperBound.branch row8111_block_012_covers row8111_block_013_covers

def row8111_large_right_right : List Nat := row8111_block_011 ++ row8111_large_right_right_right
theorem row8111_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8111 row8111_large_right_right :=
  Budget.RowUpperBound.branch row8111_block_011_covers row8111_large_right_right_right_covers

def row8111_large_right : List Nat := row8111_large_right_left ++ row8111_large_right_right
theorem row8111_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8111 row8111_large_right :=
  Budget.RowUpperBound.branch row8111_large_right_left_covers row8111_large_right_right_covers

def row8111_large : List Nat := row8111_large_left ++ row8111_large_right
theorem row8111_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8111 row8111_large :=
  Budget.RowUpperBound.branch row8111_large_left_covers row8111_large_right_covers

def row8111 : List Nat := row8111_large
theorem row8111_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8111 row8111 := row8111_large_covers

def row8117_small : List Nat := PrimePairUpper.smallRow8117
theorem row8117_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8117 row8117_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8117_covers

def row8117_block_005 : List Nat := []
def plan8117_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 37), (.secondFactor 19), (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.secondFactor 29), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 29), .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8117_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8117 row8117_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8117_block_005
  · decide
  · rfl

def row8117_block_006 : List Nat := [10607, 11261, 11987]
def plan8117_block_006 : List Budget.Choice := [(.secondFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .second, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8117_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8117 row8117_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8117_block_006
  · decide
  · rfl

def row8117_block_007 : List Nat := [13997]
def plan8117_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .second, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .second, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .second, .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8117_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8117 row8117_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8117_block_007
  · decide
  · rfl

def row8117_block_008 : List Nat := [15809]
def plan8117_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 29), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .possible, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8117_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8117 row8117_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8117_block_008
  · decide
  · rfl

def row8117_block_009 : List Nat := [16937, 17657, 17783]
def plan8117_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .possible, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8117_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8117 row8117_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8117_block_009
  · decide
  · rfl

def row8117_block_010 : List Nat := []
theorem row8117_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8117 row8117_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8117_block_011 : List Nat := []
theorem row8117_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8117 row8117_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8117_block_012 : List Nat := []
theorem row8117_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8117 row8117_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8117_block_013 : List Nat := []
theorem row8117_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8117 row8117_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8117_large_left_left : List Nat := row8117_small ++ row8117_block_005
theorem row8117_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8117 row8117_large_left_left :=
  Budget.RowUpperBound.branch row8117_small_covers row8117_block_005_covers

def row8117_large_left_right_right : List Nat := row8117_block_007 ++ row8117_block_008
theorem row8117_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8117 row8117_large_left_right_right :=
  Budget.RowUpperBound.branch row8117_block_007_covers row8117_block_008_covers

def row8117_large_left_right : List Nat := row8117_block_006 ++ row8117_large_left_right_right
theorem row8117_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8117 row8117_large_left_right :=
  Budget.RowUpperBound.branch row8117_block_006_covers row8117_large_left_right_right_covers

def row8117_large_left : List Nat := row8117_large_left_left ++ row8117_large_left_right
theorem row8117_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8117 row8117_large_left :=
  Budget.RowUpperBound.branch row8117_large_left_left_covers row8117_large_left_right_covers

def row8117_large_right_left : List Nat := row8117_block_009 ++ row8117_block_010
theorem row8117_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8117 row8117_large_right_left :=
  Budget.RowUpperBound.branch row8117_block_009_covers row8117_block_010_covers

def row8117_large_right_right_right : List Nat := row8117_block_012 ++ row8117_block_013
theorem row8117_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8117 row8117_large_right_right_right :=
  Budget.RowUpperBound.branch row8117_block_012_covers row8117_block_013_covers

def row8117_large_right_right : List Nat := row8117_block_011 ++ row8117_large_right_right_right
theorem row8117_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8117 row8117_large_right_right :=
  Budget.RowUpperBound.branch row8117_block_011_covers row8117_large_right_right_right_covers

def row8117_large_right : List Nat := row8117_large_right_left ++ row8117_large_right_right
theorem row8117_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8117 row8117_large_right :=
  Budget.RowUpperBound.branch row8117_large_right_left_covers row8117_large_right_right_covers

def row8117_large : List Nat := row8117_large_left ++ row8117_large_right
theorem row8117_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8117 row8117_large :=
  Budget.RowUpperBound.branch row8117_large_left_covers row8117_large_right_covers

def row8117 : List Nat := row8117_large
theorem row8117_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8117 row8117 := row8117_large_covers

def row8123_small : List Nat := PrimePairUpper.smallRow8123
theorem row8123_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8123 row8123_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8123_covers

def row8123_block_005 : List Nat := [9377, 9533]
def plan8123_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .possible, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 31), (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .possible, .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 37), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8123_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8123 row8123_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8123_block_005
  · decide
  · rfl

def row8123_block_006 : List Nat := [11399, 11657, 12101]
def plan8123_block_006 : List Budget.Choice := [(.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .second, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 13), .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .possible, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 29), .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8123_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8123 row8123_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8123_block_006
  · decide
  · rfl

def row8123_block_007 : List Nat := [13037, 13757, 14159]
def plan8123_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), (.secondFactor 17), .first, .first, .first, .second, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .second, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 29), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8123_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8123 row8123_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8123_block_007
  · decide
  · rfl

def row8123_block_008 : List Nat := [15269, 16061]
def plan8123_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.secondFactor 11), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8123_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8123 row8123_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8123_block_008
  · decide
  · rfl

def row8123_block_009 : List Nat := [16943, 17327]
def plan8123_block_009 : List Budget.Choice := [.first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .possible, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .possible, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8123_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8123 row8123_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8123_block_009
  · decide
  · rfl

def row8123_block_010 : List Nat := []
theorem row8123_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8123 row8123_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8123_block_011 : List Nat := []
theorem row8123_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8123 row8123_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8123_block_012 : List Nat := []
theorem row8123_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8123 row8123_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8123_block_013 : List Nat := []
theorem row8123_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8123 row8123_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8123_large_left_left : List Nat := row8123_small ++ row8123_block_005
theorem row8123_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8123 row8123_large_left_left :=
  Budget.RowUpperBound.branch row8123_small_covers row8123_block_005_covers

def row8123_large_left_right_right : List Nat := row8123_block_007 ++ row8123_block_008
theorem row8123_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8123 row8123_large_left_right_right :=
  Budget.RowUpperBound.branch row8123_block_007_covers row8123_block_008_covers

def row8123_large_left_right : List Nat := row8123_block_006 ++ row8123_large_left_right_right
theorem row8123_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8123 row8123_large_left_right :=
  Budget.RowUpperBound.branch row8123_block_006_covers row8123_large_left_right_right_covers

def row8123_large_left : List Nat := row8123_large_left_left ++ row8123_large_left_right
theorem row8123_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8123 row8123_large_left :=
  Budget.RowUpperBound.branch row8123_large_left_left_covers row8123_large_left_right_covers

def row8123_large_right_left : List Nat := row8123_block_009 ++ row8123_block_010
theorem row8123_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8123 row8123_large_right_left :=
  Budget.RowUpperBound.branch row8123_block_009_covers row8123_block_010_covers

def row8123_large_right_right_right : List Nat := row8123_block_012 ++ row8123_block_013
theorem row8123_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8123 row8123_large_right_right_right :=
  Budget.RowUpperBound.branch row8123_block_012_covers row8123_block_013_covers

def row8123_large_right_right : List Nat := row8123_block_011 ++ row8123_large_right_right_right
theorem row8123_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8123 row8123_large_right_right :=
  Budget.RowUpperBound.branch row8123_block_011_covers row8123_large_right_right_right_covers

def row8123_large_right : List Nat := row8123_large_right_left ++ row8123_large_right_right
theorem row8123_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8123 row8123_large_right :=
  Budget.RowUpperBound.branch row8123_large_right_left_covers row8123_large_right_right_covers

def row8123_large : List Nat := row8123_large_left ++ row8123_large_right
theorem row8123_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8123 row8123_large :=
  Budget.RowUpperBound.branch row8123_large_left_covers row8123_large_right_covers

def row8123 : List Nat := row8123_large
theorem row8123_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8123 row8123 := row8123_large_covers

end ProjectEulerSolutions.PrimePairLarge
