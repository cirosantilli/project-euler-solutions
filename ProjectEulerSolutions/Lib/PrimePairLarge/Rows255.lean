import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows079

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8147_small : List Nat := PrimePairUpper.smallRow8147
theorem row8147_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8147 row8147_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8147_covers

def row8147_block_005 : List Nat := [8597, 8753, 8963, 9173, 9533]
def plan8147_block_005 : List Budget.Choice := [.first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .possible, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8147_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8147 row8147_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8147_block_005
  · decide
  · rfl

def row8147_block_006 : List Nat := [10889, 11273, 11897, 12377]
def plan8147_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .second, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .possible, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 37), (.secondFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .possible, .second, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .possible, .first, .first, (.secondFactor 11), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8147_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8147 row8147_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8147_block_006
  · decide
  · rfl

def row8147_block_007 : List Nat := [12911, 13931, 14057, 14207]
def plan8147_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .possible, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 29), .first, (.secondFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .second, .first, (.secondFactor 19), .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8147_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8147 row8147_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8147_block_007
  · decide
  · rfl

def row8147_block_008 : List Nat := [14753, 14759, 14867, 15263, 15581]
def plan8147_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .possible, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 23), .second, .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .possible, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 17), .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .possible, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .second, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 11)]
set_option maxRecDepth 2048 in
theorem row8147_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8147 row8147_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8147_block_008
  · decide
  · rfl

def row8147_block_009 : List Nat := [17471, 17609]
def plan8147_block_009 : List Budget.Choice := [.first, (.firstFactor 11), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 29), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8147_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8147 row8147_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8147_block_009
  · decide
  · rfl

def row8147_block_010 : List Nat := []
theorem row8147_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8147 row8147_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8147_block_011 : List Nat := []
theorem row8147_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8147 row8147_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8147_block_012 : List Nat := []
theorem row8147_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8147 row8147_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8147_block_013 : List Nat := []
theorem row8147_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8147 row8147_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8147_large_left_left : List Nat := row8147_small ++ row8147_block_005
theorem row8147_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8147 row8147_large_left_left :=
  Budget.RowUpperBound.branch row8147_small_covers row8147_block_005_covers

def row8147_large_left_right_right : List Nat := row8147_block_007 ++ row8147_block_008
theorem row8147_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8147 row8147_large_left_right_right :=
  Budget.RowUpperBound.branch row8147_block_007_covers row8147_block_008_covers

def row8147_large_left_right : List Nat := row8147_block_006 ++ row8147_large_left_right_right
theorem row8147_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8147 row8147_large_left_right :=
  Budget.RowUpperBound.branch row8147_block_006_covers row8147_large_left_right_right_covers

def row8147_large_left : List Nat := row8147_large_left_left ++ row8147_large_left_right
theorem row8147_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8147 row8147_large_left :=
  Budget.RowUpperBound.branch row8147_large_left_left_covers row8147_large_left_right_covers

def row8147_large_right_left : List Nat := row8147_block_009 ++ row8147_block_010
theorem row8147_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8147 row8147_large_right_left :=
  Budget.RowUpperBound.branch row8147_block_009_covers row8147_block_010_covers

def row8147_large_right_right_right : List Nat := row8147_block_012 ++ row8147_block_013
theorem row8147_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8147 row8147_large_right_right_right :=
  Budget.RowUpperBound.branch row8147_block_012_covers row8147_block_013_covers

def row8147_large_right_right : List Nat := row8147_block_011 ++ row8147_large_right_right_right
theorem row8147_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8147 row8147_large_right_right :=
  Budget.RowUpperBound.branch row8147_block_011_covers row8147_large_right_right_right_covers

def row8147_large_right : List Nat := row8147_large_right_left ++ row8147_large_right_right
theorem row8147_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8147 row8147_large_right :=
  Budget.RowUpperBound.branch row8147_large_right_left_covers row8147_large_right_right_covers

def row8147_large : List Nat := row8147_large_left ++ row8147_large_right
theorem row8147_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8147 row8147_large :=
  Budget.RowUpperBound.branch row8147_large_left_covers row8147_large_right_covers

def row8147 : List Nat := row8147_large
theorem row8147_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8147 row8147 := row8147_large_covers

def row8161_small : List Nat := PrimePairUpper.smallRow8161
theorem row8161_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8161 row8161_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8161_covers

def row8161_block_005 : List Nat := [8389, 8839, 8863, 8923, 9103, 9463, 10093]
def plan8161_block_005 : List Budget.Choice := [.possible, (.secondFactor 17), .first, .first, (.secondFactor 23), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .possible, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 13), .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 17), .possible, (.firstFactor 11), (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row8161_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8161 row8161_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8161_block_005
  · decide
  · rfl

def row8161_block_006 : List Nat := [11833, 12109, 12163]
def plan8161_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), (.secondFactor 19), (.secondFactor 29), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8161_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8161 row8161_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8161_block_006
  · decide
  · rfl

def row8161_block_007 : List Nat := [12613, 13171, 13219, 14407]
def plan8161_block_007 : List Budget.Choice := [(.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 7), .second, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 31), .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), (.secondFactor 17), .first, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 19), .first, .possible, (.firstFactor 11), (.secondFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .second, .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 11), (.firstFactor 31), .possible, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 31), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8161_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8161 row8161_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8161_block_007
  · decide
  · rfl

def row8161_block_008 : List Nat := [15541, 15649, 15823, 16033]
def plan8161_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 17), .second, (.secondFactor 11), (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 11), (.secondFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8161_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8161 row8161_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8161_block_008
  · decide
  · rfl

def row8161_block_009 : List Nat := [16759, 17839]
def plan8161_block_009 : List Budget.Choice := [(.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 19), .possible, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 31), .second, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .possible, .second, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8161_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8161 row8161_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8161_block_009
  · decide
  · rfl

def row8161_block_010 : List Nat := []
theorem row8161_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8161 row8161_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8161_block_011 : List Nat := []
theorem row8161_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8161 row8161_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8161_block_012 : List Nat := []
theorem row8161_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8161 row8161_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8161_block_013 : List Nat := []
theorem row8161_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8161 row8161_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8161_large_left_left : List Nat := row8161_small ++ row8161_block_005
theorem row8161_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8161 row8161_large_left_left :=
  Budget.RowUpperBound.branch row8161_small_covers row8161_block_005_covers

def row8161_large_left_right_right : List Nat := row8161_block_007 ++ row8161_block_008
theorem row8161_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8161 row8161_large_left_right_right :=
  Budget.RowUpperBound.branch row8161_block_007_covers row8161_block_008_covers

def row8161_large_left_right : List Nat := row8161_block_006 ++ row8161_large_left_right_right
theorem row8161_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8161 row8161_large_left_right :=
  Budget.RowUpperBound.branch row8161_block_006_covers row8161_large_left_right_right_covers

def row8161_large_left : List Nat := row8161_large_left_left ++ row8161_large_left_right
theorem row8161_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8161 row8161_large_left :=
  Budget.RowUpperBound.branch row8161_large_left_left_covers row8161_large_left_right_covers

def row8161_large_right_left : List Nat := row8161_block_009 ++ row8161_block_010
theorem row8161_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8161 row8161_large_right_left :=
  Budget.RowUpperBound.branch row8161_block_009_covers row8161_block_010_covers

def row8161_large_right_right_right : List Nat := row8161_block_012 ++ row8161_block_013
theorem row8161_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8161 row8161_large_right_right_right :=
  Budget.RowUpperBound.branch row8161_block_012_covers row8161_block_013_covers

def row8161_large_right_right : List Nat := row8161_block_011 ++ row8161_large_right_right_right
theorem row8161_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8161 row8161_large_right_right :=
  Budget.RowUpperBound.branch row8161_block_011_covers row8161_large_right_right_right_covers

def row8161_large_right : List Nat := row8161_large_right_left ++ row8161_large_right_right
theorem row8161_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8161 row8161_large_right :=
  Budget.RowUpperBound.branch row8161_large_right_left_covers row8161_large_right_right_covers

def row8161_large : List Nat := row8161_large_left ++ row8161_large_right
theorem row8161_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8161 row8161_large :=
  Budget.RowUpperBound.branch row8161_large_left_covers row8161_large_right_covers

def row8161 : List Nat := row8161_large
theorem row8161_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8161 row8161 := row8161_large_covers

def row8167_small : List Nat := PrimePairUpper.smallRow8167
theorem row8167_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8167 row8167_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8167_covers

def row8167_block_005 : List Nat := [8563, 9001, 9439]
def plan8167_block_005 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .possible, .first, .second, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .possible, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 11), (.firstFactor 29), (.secondFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 23), (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8167_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8167 row8167_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8167_block_005
  · decide
  · rfl

def row8167_block_006 : List Nat := [11251, 11719, 12157, 12409]
def plan8167_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .possible, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .second, .first, .first, .second, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row8167_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8167 row8167_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8167_block_006
  · decide
  · rfl

def row8167_block_007 : List Nat := [12979, 14401]
def plan8167_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .possible, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .possible, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row8167_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8167 row8167_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8167_block_007
  · decide
  · rfl

def row8167_block_008 : List Nat := [14551, 15877]
def plan8167_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .second, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8167_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8167 row8167_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8167_block_008
  · decide
  · rfl

def row8167_block_009 : List Nat := [16903, 16963, 16981, 17659]
def plan8167_block_009 : List Budget.Choice := [(.secondFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .second, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .possible, .second, (.firstFactor 13), .first, .first, .first, .possible, .first, .possible, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .second, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8167_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8167 row8167_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8167_block_009
  · decide
  · rfl

def row8167_block_010 : List Nat := []
theorem row8167_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8167 row8167_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8167_block_011 : List Nat := []
theorem row8167_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8167 row8167_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8167_block_012 : List Nat := []
theorem row8167_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8167 row8167_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8167_block_013 : List Nat := []
theorem row8167_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8167 row8167_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8167_large_left_left : List Nat := row8167_small ++ row8167_block_005
theorem row8167_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8167 row8167_large_left_left :=
  Budget.RowUpperBound.branch row8167_small_covers row8167_block_005_covers

def row8167_large_left_right_right : List Nat := row8167_block_007 ++ row8167_block_008
theorem row8167_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8167 row8167_large_left_right_right :=
  Budget.RowUpperBound.branch row8167_block_007_covers row8167_block_008_covers

def row8167_large_left_right : List Nat := row8167_block_006 ++ row8167_large_left_right_right
theorem row8167_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8167 row8167_large_left_right :=
  Budget.RowUpperBound.branch row8167_block_006_covers row8167_large_left_right_right_covers

def row8167_large_left : List Nat := row8167_large_left_left ++ row8167_large_left_right
theorem row8167_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8167 row8167_large_left :=
  Budget.RowUpperBound.branch row8167_large_left_left_covers row8167_large_left_right_covers

def row8167_large_right_left : List Nat := row8167_block_009 ++ row8167_block_010
theorem row8167_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8167 row8167_large_right_left :=
  Budget.RowUpperBound.branch row8167_block_009_covers row8167_block_010_covers

def row8167_large_right_right_right : List Nat := row8167_block_012 ++ row8167_block_013
theorem row8167_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8167 row8167_large_right_right_right :=
  Budget.RowUpperBound.branch row8167_block_012_covers row8167_block_013_covers

def row8167_large_right_right : List Nat := row8167_block_011 ++ row8167_large_right_right_right
theorem row8167_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8167 row8167_large_right_right :=
  Budget.RowUpperBound.branch row8167_block_011_covers row8167_large_right_right_right_covers

def row8167_large_right : List Nat := row8167_large_right_left ++ row8167_large_right_right
theorem row8167_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8167 row8167_large_right :=
  Budget.RowUpperBound.branch row8167_large_right_left_covers row8167_large_right_right_covers

def row8167_large : List Nat := row8167_large_left ++ row8167_large_right
theorem row8167_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8167 row8167_large :=
  Budget.RowUpperBound.branch row8167_large_left_covers row8167_large_right_covers

def row8167 : List Nat := row8167_large
theorem row8167_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8167 row8167 := row8167_large_covers

def row8171_small : List Nat := PrimePairUpper.smallRow8171
theorem row8171_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8171 row8171_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow8171_covers

def row8171_block_005 : List Nat := [8537, 8807]
def plan8171_block_005 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 29), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 17), .second, (.secondFactor 7), .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 13), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8171_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8171 row8171_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8171_block_005
  · decide
  · rfl

def row8171_block_006 : List Nat := [10589, 10631, 10709, 11597, 12041, 12143, 12269]
def plan8171_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 17), (.firstFactor 13), .second, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .possible, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 11), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8171_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8171 row8171_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8171_block_006
  · decide
  · rfl

def row8171_block_007 : List Nat := [12899, 13421, 14387]
def plan8171_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.firstFactor 29), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 31), (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 11), .second, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8171_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8171 row8171_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8171_block_007
  · decide
  · rfl

def row8171_block_008 : List Nat := [14657, 15581, 15737]
def plan8171_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .second, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 31), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 29), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .second, (.firstFactor 7), (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8171_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8171 row8171_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8171_block_008
  · decide
  · rfl

def row8171_block_009 : List Nat := [17117]
def plan8171_block_009 : List Budget.Choice := [.first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8171_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8171 row8171_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8171_block_009
  · decide
  · rfl

def row8171_block_010 : List Nat := []
theorem row8171_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8171 row8171_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8171_block_011 : List Nat := []
theorem row8171_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8171 row8171_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8171_block_012 : List Nat := []
theorem row8171_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8171 row8171_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8171_block_013 : List Nat := []
theorem row8171_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8171 row8171_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8171_large_left_left : List Nat := row8171_small ++ row8171_block_005
theorem row8171_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8171 row8171_large_left_left :=
  Budget.RowUpperBound.branch row8171_small_covers row8171_block_005_covers

def row8171_large_left_right_right : List Nat := row8171_block_007 ++ row8171_block_008
theorem row8171_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8171 row8171_large_left_right_right :=
  Budget.RowUpperBound.branch row8171_block_007_covers row8171_block_008_covers

def row8171_large_left_right : List Nat := row8171_block_006 ++ row8171_large_left_right_right
theorem row8171_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8171 row8171_large_left_right :=
  Budget.RowUpperBound.branch row8171_block_006_covers row8171_large_left_right_right_covers

def row8171_large_left : List Nat := row8171_large_left_left ++ row8171_large_left_right
theorem row8171_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8171 row8171_large_left :=
  Budget.RowUpperBound.branch row8171_large_left_left_covers row8171_large_left_right_covers

def row8171_large_right_left : List Nat := row8171_block_009 ++ row8171_block_010
theorem row8171_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8171 row8171_large_right_left :=
  Budget.RowUpperBound.branch row8171_block_009_covers row8171_block_010_covers

def row8171_large_right_right_right : List Nat := row8171_block_012 ++ row8171_block_013
theorem row8171_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8171 row8171_large_right_right_right :=
  Budget.RowUpperBound.branch row8171_block_012_covers row8171_block_013_covers

def row8171_large_right_right : List Nat := row8171_block_011 ++ row8171_large_right_right_right
theorem row8171_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8171 row8171_large_right_right :=
  Budget.RowUpperBound.branch row8171_block_011_covers row8171_large_right_right_right_covers

def row8171_large_right : List Nat := row8171_large_right_left ++ row8171_large_right_right
theorem row8171_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8171 row8171_large_right :=
  Budget.RowUpperBound.branch row8171_large_right_left_covers row8171_large_right_right_covers

def row8171_large : List Nat := row8171_large_left ++ row8171_large_right
theorem row8171_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8171 row8171_large :=
  Budget.RowUpperBound.branch row8171_large_left_covers row8171_large_right_covers

def row8171 : List Nat := row8171_large
theorem row8171_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8171 row8171 := row8171_large_covers

end ProjectEulerSolutions.PrimePairLarge
