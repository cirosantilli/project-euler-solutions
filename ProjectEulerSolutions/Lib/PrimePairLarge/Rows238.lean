import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds
import ProjectEulerSolutions.Lib.PrimePairUpper.SmallRows071

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row7537_small : List Nat := PrimePairUpper.smallRow7537
theorem row7537_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7537 row7537_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7537_covers

def row7537_block_005 : List Nat := [8419, 8713, 9781, 9949, 10267, 10303]
def plan7537_block_005 : List Budget.Choice := [(.secondFactor 23), .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .possible, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 13), .second, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 13), .second, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7537_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7537 row7537_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7537_block_005
  · decide
  · rfl

def row7537_block_006 : List Nat := [10429, 10729, 11689, 11821, 12097]
def plan7537_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 11), .possible, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, .possible, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7537_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7537 row7537_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7537_block_006
  · decide
  · rfl

def row7537_block_007 : List Nat := [13327]
def plan7537_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 17), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .second, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row7537_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7537 row7537_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7537_block_007
  · decide
  · rfl

def row7537_block_008 : List Nat := [14869, 15241, 15661, 15667, 16057, 16273]
def plan7537_block_008 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .possible, .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 31), .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .possible, .possible, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .possible, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7537_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7537 row7537_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7537_block_008
  · decide
  · rfl

def row7537_block_009 : List Nat := [17299, 17989]
def plan7537_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 17), (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .possible, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 7), .second, (.secondFactor 19), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 7), (.secondFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 37), (.secondFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7537_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7537 row7537_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7537_block_009
  · decide
  · rfl

def row7537_block_010 : List Nat := []
theorem row7537_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7537 row7537_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7537_block_011 : List Nat := []
theorem row7537_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7537 row7537_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7537_block_012 : List Nat := []
theorem row7537_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7537 row7537_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7537_block_013 : List Nat := []
theorem row7537_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7537 row7537_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7537_large_left_left : List Nat := row7537_small ++ row7537_block_005
theorem row7537_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7537 row7537_large_left_left :=
  Budget.RowUpperBound.branch row7537_small_covers row7537_block_005_covers

def row7537_large_left_right_right : List Nat := row7537_block_007 ++ row7537_block_008
theorem row7537_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7537 row7537_large_left_right_right :=
  Budget.RowUpperBound.branch row7537_block_007_covers row7537_block_008_covers

def row7537_large_left_right : List Nat := row7537_block_006 ++ row7537_large_left_right_right
theorem row7537_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7537 row7537_large_left_right :=
  Budget.RowUpperBound.branch row7537_block_006_covers row7537_large_left_right_right_covers

def row7537_large_left : List Nat := row7537_large_left_left ++ row7537_large_left_right
theorem row7537_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7537 row7537_large_left :=
  Budget.RowUpperBound.branch row7537_large_left_left_covers row7537_large_left_right_covers

def row7537_large_right_left : List Nat := row7537_block_009 ++ row7537_block_010
theorem row7537_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7537 row7537_large_right_left :=
  Budget.RowUpperBound.branch row7537_block_009_covers row7537_block_010_covers

def row7537_large_right_right_right : List Nat := row7537_block_012 ++ row7537_block_013
theorem row7537_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7537 row7537_large_right_right_right :=
  Budget.RowUpperBound.branch row7537_block_012_covers row7537_block_013_covers

def row7537_large_right_right : List Nat := row7537_block_011 ++ row7537_large_right_right_right
theorem row7537_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7537 row7537_large_right_right :=
  Budget.RowUpperBound.branch row7537_block_011_covers row7537_large_right_right_right_covers

def row7537_large_right : List Nat := row7537_large_right_left ++ row7537_large_right_right
theorem row7537_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7537 row7537_large_right :=
  Budget.RowUpperBound.branch row7537_large_right_left_covers row7537_large_right_right_covers

def row7537_large : List Nat := row7537_large_left ++ row7537_large_right
theorem row7537_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7537 row7537_large :=
  Budget.RowUpperBound.branch row7537_large_left_covers row7537_large_right_covers

def row7537 : List Nat := row7537_large
theorem row7537_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7537 row7537 := row7537_large_covers

def row7541_small : List Nat := PrimePairUpper.smallRow7541
theorem row7541_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7541 row7541_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7541_covers

def row7541_block_005 : List Nat := [8423, 8699, 8747, 9011, 9857, 9941]
def plan7541_block_005 : List Budget.Choice := [.first, .first, .possible, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .possible, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 37), (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, (.firstFactor 37), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 31), .second, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row7541_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7541 row7541_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7541_block_005
  · decide
  · rfl

def row7541_block_006 : List Nat := [10457, 11471]
def plan7541_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 23), .second, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .possible, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row7541_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7541 row7541_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7541_block_006
  · decide
  · rfl

def row7541_block_007 : List Nat := []
def plan7541_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 31), .first, (.firstFactor 17), .second, .first, .second, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 13), .first, .second, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row7541_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7541 row7541_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7541_block_007
  · decide
  · rfl

def row7541_block_008 : List Nat := [15443, 15497]
def plan7541_block_008 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .second, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 17), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 31), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, (.secondFactor 29), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .second, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row7541_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7541 row7541_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7541_block_008
  · decide
  · rfl

def row7541_block_009 : List Nat := [17393, 17981]
def plan7541_block_009 : List Budget.Choice := [.first, .second, (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .possible, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .second, (.secondFactor 11), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7541_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7541 row7541_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7541_block_009
  · decide
  · rfl

def row7541_block_010 : List Nat := []
theorem row7541_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7541 row7541_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7541_block_011 : List Nat := []
theorem row7541_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7541 row7541_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7541_block_012 : List Nat := []
theorem row7541_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7541 row7541_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7541_block_013 : List Nat := []
theorem row7541_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7541 row7541_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7541_large_left_left : List Nat := row7541_small ++ row7541_block_005
theorem row7541_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7541 row7541_large_left_left :=
  Budget.RowUpperBound.branch row7541_small_covers row7541_block_005_covers

def row7541_large_left_right_right : List Nat := row7541_block_007 ++ row7541_block_008
theorem row7541_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7541 row7541_large_left_right_right :=
  Budget.RowUpperBound.branch row7541_block_007_covers row7541_block_008_covers

def row7541_large_left_right : List Nat := row7541_block_006 ++ row7541_large_left_right_right
theorem row7541_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7541 row7541_large_left_right :=
  Budget.RowUpperBound.branch row7541_block_006_covers row7541_large_left_right_right_covers

def row7541_large_left : List Nat := row7541_large_left_left ++ row7541_large_left_right
theorem row7541_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7541 row7541_large_left :=
  Budget.RowUpperBound.branch row7541_large_left_left_covers row7541_large_left_right_covers

def row7541_large_right_left : List Nat := row7541_block_009 ++ row7541_block_010
theorem row7541_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7541 row7541_large_right_left :=
  Budget.RowUpperBound.branch row7541_block_009_covers row7541_block_010_covers

def row7541_large_right_right_right : List Nat := row7541_block_012 ++ row7541_block_013
theorem row7541_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7541 row7541_large_right_right_right :=
  Budget.RowUpperBound.branch row7541_block_012_covers row7541_block_013_covers

def row7541_large_right_right : List Nat := row7541_block_011 ++ row7541_large_right_right_right
theorem row7541_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7541 row7541_large_right_right :=
  Budget.RowUpperBound.branch row7541_block_011_covers row7541_large_right_right_right_covers

def row7541_large_right : List Nat := row7541_large_right_left ++ row7541_large_right_right
theorem row7541_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7541 row7541_large_right :=
  Budget.RowUpperBound.branch row7541_large_right_left_covers row7541_large_right_right_covers

def row7541_large : List Nat := row7541_large_left ++ row7541_large_right
theorem row7541_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7541 row7541_large :=
  Budget.RowUpperBound.branch row7541_large_left_covers row7541_large_right_covers

def row7541 : List Nat := row7541_large
theorem row7541_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7541 row7541 := row7541_large_covers

def row7547_small : List Nat := PrimePairUpper.smallRow7547
theorem row7547_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7547 row7547_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7547_covers

def row7547_block_005 : List Nat := [9059, 9857, 9923]
def plan7547_block_005 : List Budget.Choice := [.first, .first, (.secondFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .second, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 29), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 7), (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row7547_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7547 row7547_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7547_block_005
  · decide
  · rfl

def row7547_block_006 : List Nat := [10487, 12113, 12377]
def plan7547_block_006 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .possible, .second, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 17), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .possible, .first, .first, .second, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7547_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7547 row7547_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7547_block_006
  · decide
  · rfl

def row7547_block_007 : List Nat := [12611, 12743, 13499, 13721]
def plan7547_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, .first, .first, .second, (.firstFactor 13), .second, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 37), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row7547_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7547 row7547_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7547_block_007
  · decide
  · rfl

def row7547_block_008 : List Nat := [14669]
def plan7547_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .first, .first, .second, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 19), (.firstFactor 11), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 29), (.secondFactor 31), (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .second, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row7547_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7547 row7547_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7547_block_008
  · decide
  · rfl

def row7547_block_009 : List Nat := []
def plan7547_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 23), .first, .second, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .second, .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7547_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7547 row7547_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7547_block_009
  · decide
  · rfl

def row7547_block_010 : List Nat := []
theorem row7547_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7547 row7547_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7547_block_011 : List Nat := []
theorem row7547_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7547 row7547_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7547_block_012 : List Nat := []
theorem row7547_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7547 row7547_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7547_block_013 : List Nat := []
theorem row7547_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7547 row7547_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7547_large_left_left : List Nat := row7547_small ++ row7547_block_005
theorem row7547_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7547 row7547_large_left_left :=
  Budget.RowUpperBound.branch row7547_small_covers row7547_block_005_covers

def row7547_large_left_right_right : List Nat := row7547_block_007 ++ row7547_block_008
theorem row7547_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7547 row7547_large_left_right_right :=
  Budget.RowUpperBound.branch row7547_block_007_covers row7547_block_008_covers

def row7547_large_left_right : List Nat := row7547_block_006 ++ row7547_large_left_right_right
theorem row7547_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7547 row7547_large_left_right :=
  Budget.RowUpperBound.branch row7547_block_006_covers row7547_large_left_right_right_covers

def row7547_large_left : List Nat := row7547_large_left_left ++ row7547_large_left_right
theorem row7547_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7547 row7547_large_left :=
  Budget.RowUpperBound.branch row7547_large_left_left_covers row7547_large_left_right_covers

def row7547_large_right_left : List Nat := row7547_block_009 ++ row7547_block_010
theorem row7547_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7547 row7547_large_right_left :=
  Budget.RowUpperBound.branch row7547_block_009_covers row7547_block_010_covers

def row7547_large_right_right_right : List Nat := row7547_block_012 ++ row7547_block_013
theorem row7547_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7547 row7547_large_right_right_right :=
  Budget.RowUpperBound.branch row7547_block_012_covers row7547_block_013_covers

def row7547_large_right_right : List Nat := row7547_block_011 ++ row7547_large_right_right_right
theorem row7547_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7547 row7547_large_right_right :=
  Budget.RowUpperBound.branch row7547_block_011_covers row7547_large_right_right_right_covers

def row7547_large_right : List Nat := row7547_large_right_left ++ row7547_large_right_right
theorem row7547_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7547 row7547_large_right :=
  Budget.RowUpperBound.branch row7547_large_right_left_covers row7547_large_right_right_covers

def row7547_large : List Nat := row7547_large_left ++ row7547_large_right
theorem row7547_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7547 row7547_large :=
  Budget.RowUpperBound.branch row7547_large_left_covers row7547_large_right_covers

def row7547 : List Nat := row7547_large
theorem row7547_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7547 row7547 := row7547_large_covers

def row7549_small : List Nat := PrimePairUpper.smallRow7549
theorem row7549_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 7549 row7549_small :=
  Budget.RowUpperBound.of_unbounded PrimePairUpper.smallRow7549_covers

def row7549_block_005 : List Nat := [8893, 8923, 9049, 9349, 9511, 9859, 9967]
def plan7549_block_005 : List Budget.Choice := [(.firstFactor 37), (.firstFactor 19), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .possible, .possible, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, .second, .second, .first, .first, .first, .first, (.firstFactor 31), .possible, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .second, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 37), (.secondFactor 13), .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, (.firstFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row7549_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 7549 row7549_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan7549_block_005
  · decide
  · rfl

def row7549_block_006 : List Nat := [10477, 11923, 12157, 12289, 12409]
def plan7549_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 13), .first, .first, (.secondFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 23), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row7549_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 7549 row7549_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan7549_block_006
  · decide
  · rfl

def row7549_block_007 : List Nat := [12799]
def plan7549_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .second, (.secondFactor 29), .first, (.secondFactor 19), .second, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .second, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 31), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row7549_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 7549 row7549_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan7549_block_007
  · decide
  · rfl

def row7549_block_008 : List Nat := [14653, 14851]
def plan7549_block_008 : List Budget.Choice := [(.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.secondFactor 11), (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, (.secondFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row7549_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 7549 row7549_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan7549_block_008
  · decide
  · rfl

def row7549_block_009 : List Nat := [17467, 18181, 18427]
def plan7549_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .possible, .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .possible, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 23), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .possible, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row7549_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 7549 row7549_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan7549_block_009
  · decide
  · rfl

def row7549_block_010 : List Nat := []
theorem row7549_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 7549 row7549_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row7549_block_011 : List Nat := []
theorem row7549_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 7549 row7549_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row7549_block_012 : List Nat := []
theorem row7549_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 7549 row7549_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row7549_block_013 : List Nat := []
theorem row7549_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 7549 row7549_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row7549_large_left_left : List Nat := row7549_small ++ row7549_block_005
theorem row7549_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 7549 row7549_large_left_left :=
  Budget.RowUpperBound.branch row7549_small_covers row7549_block_005_covers

def row7549_large_left_right_right : List Nat := row7549_block_007 ++ row7549_block_008
theorem row7549_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 7549 row7549_large_left_right_right :=
  Budget.RowUpperBound.branch row7549_block_007_covers row7549_block_008_covers

def row7549_large_left_right : List Nat := row7549_block_006 ++ row7549_large_left_right_right
theorem row7549_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 7549 row7549_large_left_right :=
  Budget.RowUpperBound.branch row7549_block_006_covers row7549_large_left_right_right_covers

def row7549_large_left : List Nat := row7549_large_left_left ++ row7549_large_left_right
theorem row7549_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 7549 row7549_large_left :=
  Budget.RowUpperBound.branch row7549_large_left_left_covers row7549_large_left_right_covers

def row7549_large_right_left : List Nat := row7549_block_009 ++ row7549_block_010
theorem row7549_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 7549 row7549_large_right_left :=
  Budget.RowUpperBound.branch row7549_block_009_covers row7549_block_010_covers

def row7549_large_right_right_right : List Nat := row7549_block_012 ++ row7549_block_013
theorem row7549_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 7549 row7549_large_right_right_right :=
  Budget.RowUpperBound.branch row7549_block_012_covers row7549_block_013_covers

def row7549_large_right_right : List Nat := row7549_block_011 ++ row7549_large_right_right_right
theorem row7549_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 7549 row7549_large_right_right :=
  Budget.RowUpperBound.branch row7549_block_011_covers row7549_large_right_right_right_covers

def row7549_large_right : List Nat := row7549_large_right_left ++ row7549_large_right_right
theorem row7549_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 7549 row7549_large_right :=
  Budget.RowUpperBound.branch row7549_large_right_left_covers row7549_large_right_right_covers

def row7549_large : List Nat := row7549_large_left ++ row7549_large_right
theorem row7549_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 7549 row7549_large :=
  Budget.RowUpperBound.branch row7549_large_left_covers row7549_large_right_covers

def row7549 : List Nat := row7549_large
theorem row7549_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 7549 row7549 := row7549_large_covers

end ProjectEulerSolutions.PrimePairLarge
