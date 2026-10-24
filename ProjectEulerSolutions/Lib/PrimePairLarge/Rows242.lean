import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows073

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7649_small : List Nat := PrimePairUpper.smallRow7649
theorem row7649_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7649 row7649_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7649_covers

def row7649_block_005 : List Nat := [8423, 9257, 9539, 10169]
def plan7649_block_005 : List Budget.Choice := [.first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), (.firstFactor 37), .first, .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .second, (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 29), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 31), (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7649_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7649 row7649_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7649_block_005
  · decide
  · rfl

def row7649_block_006 : List Nat := [10499, 11003, 11831]
def plan7649_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, .first, .second, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 7), .first, .second, .second, (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7649_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7649 row7649_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7649_block_006
  · decide
  · rfl

def row7649_block_007 : List Nat := [12641, 13841]
def plan7649_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .possible, .second, .first, (.firstFactor 7), .first, .second, .second, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .second, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row7649_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7649 row7649_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7649_block_007
  · decide
  · rfl

def row7649_block_008 : List Nat := [14537, 15377, 15401, 15773, 15881, 16253]
def plan7649_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 23), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 23), (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), .possible, .first, .first, .second, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row7649_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7649 row7649_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7649_block_008
  · decide
  · rfl

def row7649_block_009 : List Nat := [16619, 17393, 18059, 18269]
def plan7649_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 29), .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7649_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7649 row7649_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7649_block_009
  · decide
  · rfl

def row7649_block_010 : List Nat := []
theorem row7649_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7649 row7649_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7649_block_011 : List Nat := []
theorem row7649_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7649 row7649_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7649_block_012 : List Nat := []
theorem row7649_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7649 row7649_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7649_block_013 : List Nat := []
theorem row7649_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7649 row7649_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7649_large_left_left : List Nat := row7649_small ++ row7649_block_005
theorem row7649_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7649 row7649_large_left_left :=
  Budget.RowUpperBound.branch row7649_small_covers row7649_block_005_covers

def row7649_large_left_right_right : List Nat := row7649_block_007 ++ row7649_block_008
theorem row7649_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7649 row7649_large_left_right_right :=
  Budget.RowUpperBound.branch row7649_block_007_covers row7649_block_008_covers

def row7649_large_left_right : List Nat := row7649_block_006 ++ row7649_large_left_right_right
theorem row7649_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7649 row7649_large_left_right :=
  Budget.RowUpperBound.branch row7649_block_006_covers row7649_large_left_right_right_covers

def row7649_large_left : List Nat := row7649_large_left_left ++ row7649_large_left_right
theorem row7649_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7649 row7649_large_left :=
  Budget.RowUpperBound.branch row7649_large_left_left_covers row7649_large_left_right_covers

def row7649_large_right_left : List Nat := row7649_block_009 ++ row7649_block_010
theorem row7649_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7649 row7649_large_right_left :=
  Budget.RowUpperBound.branch row7649_block_009_covers row7649_block_010_covers

def row7649_large_right_right_right : List Nat := row7649_block_012 ++ row7649_block_013
theorem row7649_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7649 row7649_large_right_right_right :=
  Budget.RowUpperBound.branch row7649_block_012_covers row7649_block_013_covers

def row7649_large_right_right : List Nat := row7649_block_011 ++ row7649_large_right_right_right
theorem row7649_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7649 row7649_large_right_right :=
  Budget.RowUpperBound.branch row7649_block_011_covers row7649_large_right_right_right_covers

def row7649_large_right : List Nat := row7649_large_right_left ++ row7649_large_right_right
theorem row7649_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7649 row7649_large_right :=
  Budget.RowUpperBound.branch row7649_large_right_left_covers row7649_large_right_right_covers

def row7649_large : List Nat := row7649_large_left ++ row7649_large_right
theorem row7649_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7649 row7649_large :=
  Budget.RowUpperBound.branch row7649_large_left_covers row7649_large_right_covers

def row7649 : List Nat := row7649_large
theorem row7649_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7649 row7649 := row7649_large_covers

def row7669_small : List Nat := PrimePairUpper.smallRow7669
theorem row7669_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7669 row7669_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7669_covers

def row7669_block_005 : List Nat := [8803, 8887, 9433]
def plan7669_block_005 : List Budget.Choice := [(.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .second, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 13), (.secondFactor 31), .second, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .second, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .second, (.secondFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 29), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .second, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row7669_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7669 row7669_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7669_block_005
  · decide
  · rfl

def row7669_block_006 : List Nat := [11071, 11239, 11329, 11491, 11833, 11863, 12343]
def plan7669_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 11), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 11), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .possible, (.firstFactor 19), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row7669_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7669 row7669_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7669_block_006
  · decide
  · rfl

def row7669_block_007 : List Nat := [12547, 12589, 12799, 12829, 12889, 13291, 13597, 13723, 13921]
def plan7669_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 13), .possible, (.firstFactor 11), .first, (.secondFactor 29), (.secondFactor 23), .possible, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 17), .possible, .second, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .possible, (.secondFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .possible, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, .second, .first, (.firstFactor 31), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .possible, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row7669_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7669 row7669_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7669_block_007
  · decide
  · rfl

def row7669_block_008 : List Nat := [14821, 14887, 15439, 16417]
def plan7669_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .second, .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .second, (.secondFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .possible, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7669_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7669 row7669_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7669_block_008
  · decide
  · rfl

def row7669_block_009 : List Nat := [16963, 17419, 17707, 17761, 17929, 18211]
def plan7669_block_009 : List Budget.Choice := [(.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 19), (.secondFactor 23), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .possible, .first, .first, (.firstFactor 31), (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .possible, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7669_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7669 row7669_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7669_block_009
  · decide
  · rfl

def row7669_block_010 : List Nat := []
theorem row7669_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7669 row7669_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7669_block_011 : List Nat := []
theorem row7669_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7669 row7669_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7669_block_012 : List Nat := []
theorem row7669_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7669 row7669_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7669_block_013 : List Nat := []
theorem row7669_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7669 row7669_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7669_large_left_left : List Nat := row7669_small ++ row7669_block_005
theorem row7669_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7669 row7669_large_left_left :=
  Budget.RowUpperBound.branch row7669_small_covers row7669_block_005_covers

def row7669_large_left_right_right : List Nat := row7669_block_007 ++ row7669_block_008
theorem row7669_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7669 row7669_large_left_right_right :=
  Budget.RowUpperBound.branch row7669_block_007_covers row7669_block_008_covers

def row7669_large_left_right : List Nat := row7669_block_006 ++ row7669_large_left_right_right
theorem row7669_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7669 row7669_large_left_right :=
  Budget.RowUpperBound.branch row7669_block_006_covers row7669_large_left_right_right_covers

def row7669_large_left : List Nat := row7669_large_left_left ++ row7669_large_left_right
theorem row7669_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7669 row7669_large_left :=
  Budget.RowUpperBound.branch row7669_large_left_left_covers row7669_large_left_right_covers

def row7669_large_right_left : List Nat := row7669_block_009 ++ row7669_block_010
theorem row7669_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7669 row7669_large_right_left :=
  Budget.RowUpperBound.branch row7669_block_009_covers row7669_block_010_covers

def row7669_large_right_right_right : List Nat := row7669_block_012 ++ row7669_block_013
theorem row7669_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7669 row7669_large_right_right_right :=
  Budget.RowUpperBound.branch row7669_block_012_covers row7669_block_013_covers

def row7669_large_right_right : List Nat := row7669_block_011 ++ row7669_large_right_right_right
theorem row7669_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7669 row7669_large_right_right :=
  Budget.RowUpperBound.branch row7669_block_011_covers row7669_large_right_right_right_covers

def row7669_large_right : List Nat := row7669_large_right_left ++ row7669_large_right_right
theorem row7669_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7669 row7669_large_right :=
  Budget.RowUpperBound.branch row7669_large_right_left_covers row7669_large_right_right_covers

def row7669_large : List Nat := row7669_large_left ++ row7669_large_right
theorem row7669_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7669 row7669_large :=
  Budget.RowUpperBound.branch row7669_large_left_covers row7669_large_right_covers

def row7669 : List Nat := row7669_large
theorem row7669_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7669 row7669 := row7669_large_covers

def row7673_small : List Nat := PrimePairUpper.smallRow7673
theorem row7673_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7673 row7673_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7673_covers

def row7673_block_005 : List Nat := [8819, 9851, 10007]
def plan7673_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 19), .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .first, .second, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row7673_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7673 row7673_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7673_block_005
  · decide
  · rfl

def row7673_block_006 : List Nat := [11549, 12071]
def plan7673_block_006 : List Budget.Choice := [(.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7673_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7673 row7673_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7673_block_006
  · decide
  · rfl

def row7673_block_007 : List Nat := [13121, 13187, 14081, 14087]
def plan7673_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .possible, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 31), (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.secondFactor 31), .first, .second, .first, .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .possible, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row7673_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7673 row7673_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7673_block_007
  · decide
  · rfl

def row7673_block_008 : List Nat := [15359, 16127]
def plan7673_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 37), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 19), (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), .first, .possible, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7673_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7673 row7673_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7673_block_008
  · decide
  · rfl

def row7673_block_009 : List Nat := [17027, 17909]
def plan7673_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .second, .first, (.firstFactor 11), .first, .first, (.secondFactor 29), (.secondFactor 19), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 17), (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .possible, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .second, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7673_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7673 row7673_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7673_block_009
  · decide
  · rfl

def row7673_block_010 : List Nat := []
theorem row7673_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7673 row7673_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7673_block_011 : List Nat := []
theorem row7673_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7673 row7673_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7673_block_012 : List Nat := []
theorem row7673_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7673 row7673_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7673_block_013 : List Nat := []
theorem row7673_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7673 row7673_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7673_large_left_left : List Nat := row7673_small ++ row7673_block_005
theorem row7673_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7673 row7673_large_left_left :=
  Budget.RowUpperBound.branch row7673_small_covers row7673_block_005_covers

def row7673_large_left_right_right : List Nat := row7673_block_007 ++ row7673_block_008
theorem row7673_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7673 row7673_large_left_right_right :=
  Budget.RowUpperBound.branch row7673_block_007_covers row7673_block_008_covers

def row7673_large_left_right : List Nat := row7673_block_006 ++ row7673_large_left_right_right
theorem row7673_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7673 row7673_large_left_right :=
  Budget.RowUpperBound.branch row7673_block_006_covers row7673_large_left_right_right_covers

def row7673_large_left : List Nat := row7673_large_left_left ++ row7673_large_left_right
theorem row7673_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7673 row7673_large_left :=
  Budget.RowUpperBound.branch row7673_large_left_left_covers row7673_large_left_right_covers

def row7673_large_right_left : List Nat := row7673_block_009 ++ row7673_block_010
theorem row7673_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7673 row7673_large_right_left :=
  Budget.RowUpperBound.branch row7673_block_009_covers row7673_block_010_covers

def row7673_large_right_right_right : List Nat := row7673_block_012 ++ row7673_block_013
theorem row7673_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7673 row7673_large_right_right_right :=
  Budget.RowUpperBound.branch row7673_block_012_covers row7673_block_013_covers

def row7673_large_right_right : List Nat := row7673_block_011 ++ row7673_large_right_right_right
theorem row7673_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7673 row7673_large_right_right :=
  Budget.RowUpperBound.branch row7673_block_011_covers row7673_large_right_right_right_covers

def row7673_large_right : List Nat := row7673_large_right_left ++ row7673_large_right_right
theorem row7673_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7673 row7673_large_right :=
  Budget.RowUpperBound.branch row7673_large_right_left_covers row7673_large_right_right_covers

def row7673_large : List Nat := row7673_large_left ++ row7673_large_right
theorem row7673_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7673 row7673_large :=
  Budget.RowUpperBound.branch row7673_large_left_covers row7673_large_right_covers

def row7673 : List Nat := row7673_large
theorem row7673_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7673 row7673 := row7673_large_covers

def row7681_small : List Nat := PrimePairUpper.smallRow7681
theorem row7681_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7681 row7681_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7681_covers

def row7681_block_005 : List Nat := [8719, 9091, 9739]
def plan7681_block_005 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 31), .possible, (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .second, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .possible, (.firstFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .second, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, .second, .first, (.firstFactor 11), (.firstFactor 37), .possible, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.secondFactor 19), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7681_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7681 row7681_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7681_block_005
  · decide
  · rfl

def row7681_block_006 : List Nat := [10627, 11251, 11593, 12073]
def plan7681_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .second, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .possible, (.secondFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, .second, .first, .second, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row7681_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7681 row7681_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7681_block_006
  · decide
  · rfl

def row7681_block_007 : List Nat := [12739, 13183, 13249, 13921]
def plan7681_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 29), (.firstFactor 19), .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 17), .possible, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row7681_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7681 row7681_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7681_block_007
  · decide
  · rfl

def row7681_block_008 : List Nat := [14779, 15541, 15919, 16231]
def plan7681_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 23), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 23), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .possible, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 37), .first, .first, .first, .first, .possible, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 37), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7681_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7681 row7681_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7681_block_008
  · decide
  · rfl

def row7681_block_009 : List Nat := [17929, 18181, 18199]
def plan7681_block_009 : List Budget.Choice := [(.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .second, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .first, .possible, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7681_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7681 row7681_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7681_block_009
  · decide
  · rfl

def row7681_block_010 : List Nat := []
theorem row7681_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7681 row7681_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7681_block_011 : List Nat := []
theorem row7681_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7681 row7681_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7681_block_012 : List Nat := []
theorem row7681_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7681 row7681_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7681_block_013 : List Nat := []
theorem row7681_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7681 row7681_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7681_large_left_left : List Nat := row7681_small ++ row7681_block_005
theorem row7681_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7681 row7681_large_left_left :=
  Budget.RowUpperBound.branch row7681_small_covers row7681_block_005_covers

def row7681_large_left_right_right : List Nat := row7681_block_007 ++ row7681_block_008
theorem row7681_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7681 row7681_large_left_right_right :=
  Budget.RowUpperBound.branch row7681_block_007_covers row7681_block_008_covers

def row7681_large_left_right : List Nat := row7681_block_006 ++ row7681_large_left_right_right
theorem row7681_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7681 row7681_large_left_right :=
  Budget.RowUpperBound.branch row7681_block_006_covers row7681_large_left_right_right_covers

def row7681_large_left : List Nat := row7681_large_left_left ++ row7681_large_left_right
theorem row7681_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7681 row7681_large_left :=
  Budget.RowUpperBound.branch row7681_large_left_left_covers row7681_large_left_right_covers

def row7681_large_right_left : List Nat := row7681_block_009 ++ row7681_block_010
theorem row7681_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7681 row7681_large_right_left :=
  Budget.RowUpperBound.branch row7681_block_009_covers row7681_block_010_covers

def row7681_large_right_right_right : List Nat := row7681_block_012 ++ row7681_block_013
theorem row7681_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7681 row7681_large_right_right_right :=
  Budget.RowUpperBound.branch row7681_block_012_covers row7681_block_013_covers

def row7681_large_right_right : List Nat := row7681_block_011 ++ row7681_large_right_right_right
theorem row7681_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7681 row7681_large_right_right :=
  Budget.RowUpperBound.branch row7681_block_011_covers row7681_large_right_right_right_covers

def row7681_large_right : List Nat := row7681_large_right_left ++ row7681_large_right_right
theorem row7681_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7681 row7681_large_right :=
  Budget.RowUpperBound.branch row7681_large_right_left_covers row7681_large_right_right_covers

def row7681_large : List Nat := row7681_large_left ++ row7681_large_right
theorem row7681_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7681 row7681_large :=
  Budget.RowUpperBound.branch row7681_large_left_covers row7681_large_right_covers

def row7681 : List Nat := row7681_large
theorem row7681_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7681 row7681 := row7681_large_covers

end ProjectEulerSolutions.PrimePairLarge
