import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8563_small : List Nat := []
theorem row8563_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8563 row8563_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8563_block_005 : List Nat := [9343, 10141, 10243]
def plan8563_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 29), (.secondFactor 23), (.firstFactor 11), (.secondFactor 17), (.firstFactor 7), .second, .first, (.firstFactor 13), .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 23), .first, .first, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 11), .second, .first, (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, .possible, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8563_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8563 row8563_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8563_block_005
  · decide
  · rfl

def row8563_block_006 : List Nat := [11311, 11383, 11887, 12157]
def plan8563_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .possible, .second, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .possible, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8563_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8563 row8563_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8563_block_006
  · decide
  · rfl

def row8563_block_007 : List Nat := [14029]
def plan8563_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .second, .first, (.secondFactor 31), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row8563_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8563 row8563_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8563_block_007
  · decide
  · rfl

def row8563_block_008 : List Nat := [14551, 14653, 15091, 15583, 15667, 16183]
def plan8563_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 37), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8563_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8563 row8563_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8563_block_008
  · decide
  · rfl

def row8563_block_009 : List Nat := []
def plan8563_block_009 : List Budget.Choice := [(.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8563_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8563 row8563_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8563_block_009
  · decide
  · rfl

def row8563_block_010 : List Nat := []
theorem row8563_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8563 row8563_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8563_block_011 : List Nat := []
theorem row8563_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8563 row8563_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8563_block_012 : List Nat := []
theorem row8563_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8563 row8563_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8563_block_013 : List Nat := []
theorem row8563_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8563 row8563_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8563_large_left_left : List Nat := row8563_small ++ row8563_block_005
theorem row8563_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8563 row8563_large_left_left :=
  Budget.RowUpperBound.branch row8563_small_covers row8563_block_005_covers

def row8563_large_left_right_right : List Nat := row8563_block_007 ++ row8563_block_008
theorem row8563_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8563 row8563_large_left_right_right :=
  Budget.RowUpperBound.branch row8563_block_007_covers row8563_block_008_covers

def row8563_large_left_right : List Nat := row8563_block_006 ++ row8563_large_left_right_right
theorem row8563_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8563 row8563_large_left_right :=
  Budget.RowUpperBound.branch row8563_block_006_covers row8563_large_left_right_right_covers

def row8563_large_left : List Nat := row8563_large_left_left ++ row8563_large_left_right
theorem row8563_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8563 row8563_large_left :=
  Budget.RowUpperBound.branch row8563_large_left_left_covers row8563_large_left_right_covers

def row8563_large_right_left : List Nat := row8563_block_009 ++ row8563_block_010
theorem row8563_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8563 row8563_large_right_left :=
  Budget.RowUpperBound.branch row8563_block_009_covers row8563_block_010_covers

def row8563_large_right_right_right : List Nat := row8563_block_012 ++ row8563_block_013
theorem row8563_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8563 row8563_large_right_right_right :=
  Budget.RowUpperBound.branch row8563_block_012_covers row8563_block_013_covers

def row8563_large_right_right : List Nat := row8563_block_011 ++ row8563_large_right_right_right
theorem row8563_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8563 row8563_large_right_right :=
  Budget.RowUpperBound.branch row8563_block_011_covers row8563_large_right_right_right_covers

def row8563_large_right : List Nat := row8563_large_right_left ++ row8563_large_right_right
theorem row8563_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8563 row8563_large_right :=
  Budget.RowUpperBound.branch row8563_large_right_left_covers row8563_large_right_right_covers

def row8563_large : List Nat := row8563_large_left ++ row8563_large_right
theorem row8563_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8563 row8563_large :=
  Budget.RowUpperBound.branch row8563_large_left_covers row8563_large_right_covers

def row8563 : List Nat := row8563_large
theorem row8563_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8563 row8563 := row8563_large_covers

def row8573_small : List Nat := []
theorem row8573_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8573 row8573_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8573_block_005 : List Nat := [8753, 9161, 9497, 9539, 9623, 10079, 10163]
def plan8573_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, .first, .second, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 31), (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), .possible, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.firstFactor 37), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 17), .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 29), .first, .possible, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8573_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8573 row8573_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8573_block_005
  · decide
  · rfl

def row8573_block_006 : List Nat := [11471, 11549, 12251]
def plan8573_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, .first, (.secondFactor 11), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 31), (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .possible, .first, (.secondFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8573_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8573 row8573_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8573_block_006
  · decide
  · rfl

def row8573_block_007 : List Nat := [12641, 14411]
def plan8573_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, (.secondFactor 19), (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 19), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 31), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8573_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8573 row8573_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8573_block_007
  · decide
  · rfl

def row8573_block_008 : List Nat := [14957, 15749, 15791]
def plan8573_block_008 : List Budget.Choice := [.first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .second, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8573_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8573 row8573_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8573_block_008
  · decide
  · rfl

def row8573_block_009 : List Nat := [16871]
def plan8573_block_009 : List Budget.Choice := [.first, (.firstFactor 37), (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 31), .first, .second, (.firstFactor 13), .first, .second, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8573_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8573 row8573_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8573_block_009
  · decide
  · rfl

def row8573_block_010 : List Nat := []
theorem row8573_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8573 row8573_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8573_block_011 : List Nat := []
theorem row8573_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8573 row8573_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8573_block_012 : List Nat := []
theorem row8573_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8573 row8573_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8573_block_013 : List Nat := []
theorem row8573_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8573 row8573_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8573_large_left_left : List Nat := row8573_small ++ row8573_block_005
theorem row8573_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8573 row8573_large_left_left :=
  Budget.RowUpperBound.branch row8573_small_covers row8573_block_005_covers

def row8573_large_left_right_right : List Nat := row8573_block_007 ++ row8573_block_008
theorem row8573_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8573 row8573_large_left_right_right :=
  Budget.RowUpperBound.branch row8573_block_007_covers row8573_block_008_covers

def row8573_large_left_right : List Nat := row8573_block_006 ++ row8573_large_left_right_right
theorem row8573_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8573 row8573_large_left_right :=
  Budget.RowUpperBound.branch row8573_block_006_covers row8573_large_left_right_right_covers

def row8573_large_left : List Nat := row8573_large_left_left ++ row8573_large_left_right
theorem row8573_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8573 row8573_large_left :=
  Budget.RowUpperBound.branch row8573_large_left_left_covers row8573_large_left_right_covers

def row8573_large_right_left : List Nat := row8573_block_009 ++ row8573_block_010
theorem row8573_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8573 row8573_large_right_left :=
  Budget.RowUpperBound.branch row8573_block_009_covers row8573_block_010_covers

def row8573_large_right_right_right : List Nat := row8573_block_012 ++ row8573_block_013
theorem row8573_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8573 row8573_large_right_right_right :=
  Budget.RowUpperBound.branch row8573_block_012_covers row8573_block_013_covers

def row8573_large_right_right : List Nat := row8573_block_011 ++ row8573_large_right_right_right
theorem row8573_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8573 row8573_large_right_right :=
  Budget.RowUpperBound.branch row8573_block_011_covers row8573_large_right_right_right_covers

def row8573_large_right : List Nat := row8573_large_right_left ++ row8573_large_right_right
theorem row8573_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8573 row8573_large_right :=
  Budget.RowUpperBound.branch row8573_large_right_left_covers row8573_large_right_right_covers

def row8573_large : List Nat := row8573_large_left ++ row8573_large_right
theorem row8573_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8573 row8573_large :=
  Budget.RowUpperBound.branch row8573_large_left_covers row8573_large_right_covers

def row8573 : List Nat := row8573_large
theorem row8573_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8573 row8573 := row8573_large_covers

def row8581_small : List Nat := []
theorem row8581_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8581 row8581_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8581_block_005 : List Nat := [9127, 9157, 9241, 9931]
def plan8581_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 29), .second, (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 7), .second, (.secondFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), (.secondFactor 37), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8581_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8581 row8581_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8581_block_005
  · decide
  · rfl

def row8581_block_006 : List Nat := [10399, 11257, 11719, 11887, 12097, 12163]
def plan8581_block_006 : List Budget.Choice := [.first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 23), .first, .first, .first, .possible, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8581_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8581 row8581_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8581_block_006
  · decide
  · rfl

def row8581_block_007 : List Nat := [12907, 12919, 13297, 13339, 13411, 13933, 14011, 14071, 14143]
def plan8581_block_007 : List Budget.Choice := [(.secondFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 23), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .second, (.firstFactor 13), (.secondFactor 7), .first, .first, .possible, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .possible, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 23), .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.firstFactor 29), .first, (.secondFactor 23), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row8581_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8581 row8581_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8581_block_007
  · decide
  · rfl

def row8581_block_008 : List Nat := [15679, 15889]
def plan8581_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .second, .first, .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.firstFactor 37), .first, .first, .possible, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 23), (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8581_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8581 row8581_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8581_block_008
  · decide
  · rfl

def row8581_block_009 : List Nat := [16993]
def plan8581_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 19), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 29), .second, .second, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 37), (.secondFactor 17), (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8581_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8581 row8581_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8581_block_009
  · decide
  · rfl

def row8581_block_010 : List Nat := []
theorem row8581_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8581 row8581_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8581_block_011 : List Nat := []
theorem row8581_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8581 row8581_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8581_block_012 : List Nat := []
theorem row8581_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8581 row8581_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8581_block_013 : List Nat := []
theorem row8581_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8581 row8581_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8581_large_left_left : List Nat := row8581_small ++ row8581_block_005
theorem row8581_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8581 row8581_large_left_left :=
  Budget.RowUpperBound.branch row8581_small_covers row8581_block_005_covers

def row8581_large_left_right_right : List Nat := row8581_block_007 ++ row8581_block_008
theorem row8581_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8581 row8581_large_left_right_right :=
  Budget.RowUpperBound.branch row8581_block_007_covers row8581_block_008_covers

def row8581_large_left_right : List Nat := row8581_block_006 ++ row8581_large_left_right_right
theorem row8581_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8581 row8581_large_left_right :=
  Budget.RowUpperBound.branch row8581_block_006_covers row8581_large_left_right_right_covers

def row8581_large_left : List Nat := row8581_large_left_left ++ row8581_large_left_right
theorem row8581_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8581 row8581_large_left :=
  Budget.RowUpperBound.branch row8581_large_left_left_covers row8581_large_left_right_covers

def row8581_large_right_left : List Nat := row8581_block_009 ++ row8581_block_010
theorem row8581_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8581 row8581_large_right_left :=
  Budget.RowUpperBound.branch row8581_block_009_covers row8581_block_010_covers

def row8581_large_right_right_right : List Nat := row8581_block_012 ++ row8581_block_013
theorem row8581_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8581 row8581_large_right_right_right :=
  Budget.RowUpperBound.branch row8581_block_012_covers row8581_block_013_covers

def row8581_large_right_right : List Nat := row8581_block_011 ++ row8581_large_right_right_right
theorem row8581_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8581 row8581_large_right_right :=
  Budget.RowUpperBound.branch row8581_block_011_covers row8581_large_right_right_right_covers

def row8581_large_right : List Nat := row8581_large_right_left ++ row8581_large_right_right
theorem row8581_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8581 row8581_large_right :=
  Budget.RowUpperBound.branch row8581_large_right_left_covers row8581_large_right_right_covers

def row8581_large : List Nat := row8581_large_left ++ row8581_large_right
theorem row8581_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8581 row8581_large :=
  Budget.RowUpperBound.branch row8581_large_left_covers row8581_large_right_covers

def row8581 : List Nat := row8581_large
theorem row8581_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8581 row8581 := row8581_large_covers

def row8597_small : List Nat := []
theorem row8597_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8597 row8597_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8597_block_005 : List Nat := [9029, 9209, 9227, 9239, 9587, 9767, 10193, 10289, 10343]
def plan8597_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), .first, .first, .second, .first, .possible, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, (.firstFactor 23), .possible, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.secondFactor 29), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), .possible, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .possible, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8597_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8597 row8597_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8597_block_005
  · decide
  · rfl

def row8597_block_006 : List Nat := [11399]
def plan8597_block_006 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .second, .possible, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, (.secondFactor 13), .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .second, .first, (.secondFactor 19), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8597_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8597 row8597_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8597_block_006
  · decide
  · rfl

def row8597_block_007 : List Nat := [12671, 13187, 13259, 13451, 13457]
def plan8597_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 19), .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .possible, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .possible, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row8597_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8597 row8597_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8597_block_007
  · decide
  · rfl

def row8597_block_008 : List Nat := [14519]
def plan8597_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 17), .first, .first, .second, .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 11), .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .second, .first, (.secondFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 11), (.firstFactor 23), .first, .second, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 17), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 19), .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 37), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8597_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8597 row8597_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8597_block_008
  · decide
  · rfl

def row8597_block_009 : List Nat := [16607, 16619, 16661]
def plan8597_block_009 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .possible, .first, .first, (.firstFactor 11), .first, .first, .possible, (.firstFactor 17), (.secondFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8597_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8597 row8597_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8597_block_009
  · decide
  · rfl

def row8597_block_010 : List Nat := []
theorem row8597_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8597 row8597_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8597_block_011 : List Nat := []
theorem row8597_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8597 row8597_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8597_block_012 : List Nat := []
theorem row8597_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8597 row8597_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8597_block_013 : List Nat := []
theorem row8597_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8597 row8597_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8597_large_left_left : List Nat := row8597_small ++ row8597_block_005
theorem row8597_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8597 row8597_large_left_left :=
  Budget.RowUpperBound.branch row8597_small_covers row8597_block_005_covers

def row8597_large_left_right_right : List Nat := row8597_block_007 ++ row8597_block_008
theorem row8597_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8597 row8597_large_left_right_right :=
  Budget.RowUpperBound.branch row8597_block_007_covers row8597_block_008_covers

def row8597_large_left_right : List Nat := row8597_block_006 ++ row8597_large_left_right_right
theorem row8597_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8597 row8597_large_left_right :=
  Budget.RowUpperBound.branch row8597_block_006_covers row8597_large_left_right_right_covers

def row8597_large_left : List Nat := row8597_large_left_left ++ row8597_large_left_right
theorem row8597_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8597 row8597_large_left :=
  Budget.RowUpperBound.branch row8597_large_left_left_covers row8597_large_left_right_covers

def row8597_large_right_left : List Nat := row8597_block_009 ++ row8597_block_010
theorem row8597_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8597 row8597_large_right_left :=
  Budget.RowUpperBound.branch row8597_block_009_covers row8597_block_010_covers

def row8597_large_right_right_right : List Nat := row8597_block_012 ++ row8597_block_013
theorem row8597_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8597 row8597_large_right_right_right :=
  Budget.RowUpperBound.branch row8597_block_012_covers row8597_block_013_covers

def row8597_large_right_right : List Nat := row8597_block_011 ++ row8597_large_right_right_right
theorem row8597_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8597 row8597_large_right_right :=
  Budget.RowUpperBound.branch row8597_block_011_covers row8597_large_right_right_right_covers

def row8597_large_right : List Nat := row8597_large_right_left ++ row8597_large_right_right
theorem row8597_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8597 row8597_large_right :=
  Budget.RowUpperBound.branch row8597_large_right_left_covers row8597_large_right_right_covers

def row8597_large : List Nat := row8597_large_left ++ row8597_large_right
theorem row8597_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8597 row8597_large :=
  Budget.RowUpperBound.branch row8597_large_left_covers row8597_large_right_covers

def row8597 : List Nat := row8597_large
theorem row8597_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8597 row8597 := row8597_large_covers

end ProjectEulerSolutions.PrimePairLarge
