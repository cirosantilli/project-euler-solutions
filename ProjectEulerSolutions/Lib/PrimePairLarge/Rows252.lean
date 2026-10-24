import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows078

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8039_small : List Nat := PrimePairUpper.smallRow8039
theorem row8039_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8039 row8039_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8039_covers

def row8039_block_005 : List Nat := [8819, 10181, 10271]
def plan8039_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .second, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .possible, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row8039_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8039 row8039_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8039_block_005
  · decide
  · rfl

def row8039_block_006 : List Nat := [11483, 11699]
def plan8039_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .possible, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8039_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8039 row8039_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8039_block_006
  · decide
  · rfl

def row8039_block_007 : List Nat := [12437, 12647, 12911, 14087]
def plan8039_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 37), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .second, .first, (.firstFactor 11), (.firstFactor 13), .first, .second, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 11), .second, .first, .second, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 23), .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 37), (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8039_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8039 row8039_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8039_block_007
  · decide
  · rfl

def row8039_block_008 : List Nat := [14633, 15383, 15881, 16097, 16319, 16481]
def plan8039_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 23), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .second, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, .second, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, .possible, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, .possible, (.firstFactor 11), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row8039_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8039 row8039_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8039_block_008
  · decide
  · rfl

def row8039_block_009 : List Nat := [16901, 17159]
def plan8039_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 23), .possible, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .possible, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8039_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8039 row8039_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8039_block_009
  · decide
  · rfl

def row8039_block_010 : List Nat := []
theorem row8039_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8039 row8039_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8039_block_011 : List Nat := []
theorem row8039_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8039 row8039_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8039_block_012 : List Nat := []
theorem row8039_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8039 row8039_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8039_block_013 : List Nat := []
theorem row8039_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8039 row8039_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8039_large_left_left : List Nat := row8039_small ++ row8039_block_005
theorem row8039_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8039 row8039_large_left_left :=
  Budget.RowUpperBound.branch row8039_small_covers row8039_block_005_covers

def row8039_large_left_right_right : List Nat := row8039_block_007 ++ row8039_block_008
theorem row8039_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8039 row8039_large_left_right_right :=
  Budget.RowUpperBound.branch row8039_block_007_covers row8039_block_008_covers

def row8039_large_left_right : List Nat := row8039_block_006 ++ row8039_large_left_right_right
theorem row8039_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8039 row8039_large_left_right :=
  Budget.RowUpperBound.branch row8039_block_006_covers row8039_large_left_right_right_covers

def row8039_large_left : List Nat := row8039_large_left_left ++ row8039_large_left_right
theorem row8039_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8039 row8039_large_left :=
  Budget.RowUpperBound.branch row8039_large_left_left_covers row8039_large_left_right_covers

def row8039_large_right_left : List Nat := row8039_block_009 ++ row8039_block_010
theorem row8039_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8039 row8039_large_right_left :=
  Budget.RowUpperBound.branch row8039_block_009_covers row8039_block_010_covers

def row8039_large_right_right_right : List Nat := row8039_block_012 ++ row8039_block_013
theorem row8039_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8039 row8039_large_right_right_right :=
  Budget.RowUpperBound.branch row8039_block_012_covers row8039_block_013_covers

def row8039_large_right_right : List Nat := row8039_block_011 ++ row8039_large_right_right_right
theorem row8039_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8039 row8039_large_right_right :=
  Budget.RowUpperBound.branch row8039_block_011_covers row8039_large_right_right_right_covers

def row8039_large_right : List Nat := row8039_large_right_left ++ row8039_large_right_right
theorem row8039_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8039 row8039_large_right :=
  Budget.RowUpperBound.branch row8039_large_right_left_covers row8039_large_right_right_covers

def row8039_large : List Nat := row8039_large_left ++ row8039_large_right
theorem row8039_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8039 row8039_large :=
  Budget.RowUpperBound.branch row8039_large_left_covers row8039_large_right_covers

def row8039 : List Nat := row8039_large
theorem row8039_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8039 row8039 := row8039_large_covers

def row8053_small : List Nat := PrimePairUpper.smallRow8053
theorem row8053_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8053 row8053_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8053_covers

def row8053_block_005 : List Nat := [8431, 9103]
def plan8053_block_005 : List Budget.Choice := [(.secondFactor 19), (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 37), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row8053_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8053 row8053_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8053_block_005
  · decide
  · rfl

def row8053_block_006 : List Nat := [10861, 10939, 10993, 11491, 11587, 11779]
def plan8053_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8053_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8053 row8053_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8053_block_006
  · decide
  · rfl

def row8053_block_007 : List Nat := [12589, 12757, 13591, 14149, 14347]
def plan8053_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 17), (.secondFactor 11), .possible, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 7), .first, .possible, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .second, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8053_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8053 row8053_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8053_block_007
  · decide
  · rfl

def row8053_block_008 : List Nat := [15733, 15991]
def plan8053_block_008 : List Budget.Choice := [(.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, .second, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 23), (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8053_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8053 row8053_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8053_block_008
  · decide
  · rfl

def row8053_block_009 : List Nat := [17209, 17707]
def plan8053_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .possible, .first, (.firstFactor 23), .second, .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .second, .first, (.firstFactor 37), (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .second, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8053_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8053 row8053_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8053_block_009
  · decide
  · rfl

def row8053_block_010 : List Nat := []
theorem row8053_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8053 row8053_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8053_block_011 : List Nat := []
theorem row8053_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8053 row8053_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8053_block_012 : List Nat := []
theorem row8053_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8053 row8053_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8053_block_013 : List Nat := []
theorem row8053_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8053 row8053_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8053_large_left_left : List Nat := row8053_small ++ row8053_block_005
theorem row8053_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8053 row8053_large_left_left :=
  Budget.RowUpperBound.branch row8053_small_covers row8053_block_005_covers

def row8053_large_left_right_right : List Nat := row8053_block_007 ++ row8053_block_008
theorem row8053_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8053 row8053_large_left_right_right :=
  Budget.RowUpperBound.branch row8053_block_007_covers row8053_block_008_covers

def row8053_large_left_right : List Nat := row8053_block_006 ++ row8053_large_left_right_right
theorem row8053_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8053 row8053_large_left_right :=
  Budget.RowUpperBound.branch row8053_block_006_covers row8053_large_left_right_right_covers

def row8053_large_left : List Nat := row8053_large_left_left ++ row8053_large_left_right
theorem row8053_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8053 row8053_large_left :=
  Budget.RowUpperBound.branch row8053_large_left_left_covers row8053_large_left_right_covers

def row8053_large_right_left : List Nat := row8053_block_009 ++ row8053_block_010
theorem row8053_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8053 row8053_large_right_left :=
  Budget.RowUpperBound.branch row8053_block_009_covers row8053_block_010_covers

def row8053_large_right_right_right : List Nat := row8053_block_012 ++ row8053_block_013
theorem row8053_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8053 row8053_large_right_right_right :=
  Budget.RowUpperBound.branch row8053_block_012_covers row8053_block_013_covers

def row8053_large_right_right : List Nat := row8053_block_011 ++ row8053_large_right_right_right
theorem row8053_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8053 row8053_large_right_right :=
  Budget.RowUpperBound.branch row8053_block_011_covers row8053_large_right_right_right_covers

def row8053_large_right : List Nat := row8053_large_right_left ++ row8053_large_right_right
theorem row8053_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8053 row8053_large_right :=
  Budget.RowUpperBound.branch row8053_large_right_left_covers row8053_large_right_right_covers

def row8053_large : List Nat := row8053_large_left ++ row8053_large_right
theorem row8053_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8053 row8053_large :=
  Budget.RowUpperBound.branch row8053_large_left_covers row8053_large_right_covers

def row8053 : List Nat := row8053_large
theorem row8053_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8053 row8053 := row8053_large_covers

def row8059_small : List Nat := PrimePairUpper.smallRow8059
theorem row8059_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8059 row8059_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8059_covers

def row8059_block_005 : List Nat := [8461, 8527, 8737, 9067, 9151, 10093, 10243]
def plan8059_block_005 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 31), (.firstFactor 19), .first, (.secondFactor 37), .first, .first, .second, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 17), (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .possible, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .second, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .possible, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8059_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8059 row8059_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8059_block_005
  · decide
  · rfl

def row8059_block_006 : List Nat := [11863]
def plan8059_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8059_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8059 row8059_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8059_block_006
  · decide
  · rfl

def row8059_block_007 : List Nat := [12553, 13159, 13249, 13327, 14323]
def plan8059_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .first, .possible, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 19), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 31), (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 29), .first, (.secondFactor 19), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .possible, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 19), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8059_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8059 row8059_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8059_block_007
  · decide
  · rfl

def row8059_block_008 : List Nat := [15031, 15289, 15541, 15601, 15727, 16087]
def plan8059_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, .possible, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), (.firstFactor 11), .second, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8059_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8059 row8059_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8059_block_008
  · decide
  · rfl

def row8059_block_009 : List Nat := [16927, 17449]
def plan8059_block_009 : List Budget.Choice := [(.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .second, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8059_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8059 row8059_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8059_block_009
  · decide
  · rfl

def row8059_block_010 : List Nat := []
theorem row8059_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8059 row8059_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8059_block_011 : List Nat := []
theorem row8059_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8059 row8059_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8059_block_012 : List Nat := []
theorem row8059_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8059 row8059_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8059_block_013 : List Nat := []
theorem row8059_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8059 row8059_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8059_large_left_left : List Nat := row8059_small ++ row8059_block_005
theorem row8059_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8059 row8059_large_left_left :=
  Budget.RowUpperBound.branch row8059_small_covers row8059_block_005_covers

def row8059_large_left_right_right : List Nat := row8059_block_007 ++ row8059_block_008
theorem row8059_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8059 row8059_large_left_right_right :=
  Budget.RowUpperBound.branch row8059_block_007_covers row8059_block_008_covers

def row8059_large_left_right : List Nat := row8059_block_006 ++ row8059_large_left_right_right
theorem row8059_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8059 row8059_large_left_right :=
  Budget.RowUpperBound.branch row8059_block_006_covers row8059_large_left_right_right_covers

def row8059_large_left : List Nat := row8059_large_left_left ++ row8059_large_left_right
theorem row8059_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8059 row8059_large_left :=
  Budget.RowUpperBound.branch row8059_large_left_left_covers row8059_large_left_right_covers

def row8059_large_right_left : List Nat := row8059_block_009 ++ row8059_block_010
theorem row8059_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8059 row8059_large_right_left :=
  Budget.RowUpperBound.branch row8059_block_009_covers row8059_block_010_covers

def row8059_large_right_right_right : List Nat := row8059_block_012 ++ row8059_block_013
theorem row8059_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8059 row8059_large_right_right_right :=
  Budget.RowUpperBound.branch row8059_block_012_covers row8059_block_013_covers

def row8059_large_right_right : List Nat := row8059_block_011 ++ row8059_large_right_right_right
theorem row8059_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8059 row8059_large_right_right :=
  Budget.RowUpperBound.branch row8059_block_011_covers row8059_large_right_right_right_covers

def row8059_large_right : List Nat := row8059_large_right_left ++ row8059_large_right_right
theorem row8059_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8059 row8059_large_right :=
  Budget.RowUpperBound.branch row8059_large_right_left_covers row8059_large_right_right_covers

def row8059_large : List Nat := row8059_large_left ++ row8059_large_right
theorem row8059_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8059 row8059_large :=
  Budget.RowUpperBound.branch row8059_large_left_covers row8059_large_right_covers

def row8059 : List Nat := row8059_large
theorem row8059_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8059 row8059 := row8059_large_covers

def row8069_small : List Nat := PrimePairUpper.smallRow8069
theorem row8069_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8069 row8069_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8069_covers

def row8069_block_005 : List Nat := [8861, 9491, 9497, 9803, 10211]
def plan8069_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .possible, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), .second, (.firstFactor 7), .possible, .possible, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .second, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .possible, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 19), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8069_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8069 row8069_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8069_block_005
  · decide
  · rfl

def row8069_block_006 : List Nat := [11003, 11117]
def plan8069_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .possible, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 11), .second, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 23), (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), .first, .second, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8069_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8069 row8069_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8069_block_006
  · decide
  · rfl

def row8069_block_007 : List Nat := [13421, 13463, 13781, 14033, 14057, 14177, 14447]
def plan8069_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 19), (.firstFactor 7), .possible, (.secondFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .possible, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 31), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .second, .second, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8069_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8069 row8069_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8069_block_007
  · decide
  · rfl

def row8069_block_008 : List Nat := [15737]
def plan8069_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 23), .second, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 23), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 29), .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 29), (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .second, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8069_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8069 row8069_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8069_block_008
  · decide
  · rfl

def row8069_block_009 : List Nat := [16547, 17939]
def plan8069_block_009 : List Budget.Choice := [.first, (.firstFactor 19), .possible, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.secondFactor 29), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8069_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8069 row8069_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8069_block_009
  · decide
  · rfl

def row8069_block_010 : List Nat := []
theorem row8069_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8069 row8069_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8069_block_011 : List Nat := []
theorem row8069_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8069 row8069_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8069_block_012 : List Nat := []
theorem row8069_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8069 row8069_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8069_block_013 : List Nat := []
theorem row8069_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8069 row8069_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8069_large_left_left : List Nat := row8069_small ++ row8069_block_005
theorem row8069_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8069 row8069_large_left_left :=
  Budget.RowUpperBound.branch row8069_small_covers row8069_block_005_covers

def row8069_large_left_right_right : List Nat := row8069_block_007 ++ row8069_block_008
theorem row8069_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8069 row8069_large_left_right_right :=
  Budget.RowUpperBound.branch row8069_block_007_covers row8069_block_008_covers

def row8069_large_left_right : List Nat := row8069_block_006 ++ row8069_large_left_right_right
theorem row8069_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8069 row8069_large_left_right :=
  Budget.RowUpperBound.branch row8069_block_006_covers row8069_large_left_right_right_covers

def row8069_large_left : List Nat := row8069_large_left_left ++ row8069_large_left_right
theorem row8069_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8069 row8069_large_left :=
  Budget.RowUpperBound.branch row8069_large_left_left_covers row8069_large_left_right_covers

def row8069_large_right_left : List Nat := row8069_block_009 ++ row8069_block_010
theorem row8069_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8069 row8069_large_right_left :=
  Budget.RowUpperBound.branch row8069_block_009_covers row8069_block_010_covers

def row8069_large_right_right_right : List Nat := row8069_block_012 ++ row8069_block_013
theorem row8069_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8069 row8069_large_right_right_right :=
  Budget.RowUpperBound.branch row8069_block_012_covers row8069_block_013_covers

def row8069_large_right_right : List Nat := row8069_block_011 ++ row8069_large_right_right_right
theorem row8069_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8069 row8069_large_right_right :=
  Budget.RowUpperBound.branch row8069_block_011_covers row8069_large_right_right_right_covers

def row8069_large_right : List Nat := row8069_large_right_left ++ row8069_large_right_right
theorem row8069_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8069 row8069_large_right :=
  Budget.RowUpperBound.branch row8069_large_right_left_covers row8069_large_right_right_covers

def row8069_large : List Nat := row8069_large_left ++ row8069_large_right
theorem row8069_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8069 row8069_large :=
  Budget.RowUpperBound.branch row8069_large_left_covers row8069_large_right_covers

def row8069 : List Nat := row8069_large
theorem row8069_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8069 row8069 := row8069_large_covers

end ProjectEulerSolutions.PrimePairLarge
