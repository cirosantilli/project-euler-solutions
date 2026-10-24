import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8693_small : List Nat := []
theorem row8693_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8693 row8693_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8693_block_005 : List Nat := [8783, 9341, 9371, 10067]
def plan8693_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), (.secondFactor 13), (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .second, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 29), .possible, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8693_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8693 row8693_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8693_block_005
  · decide
  · rfl

def row8693_block_006 : List Nat := []
def plan8693_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 29), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 37), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 19), (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 19), .first, .first, (.secondFactor 31), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8693_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8693 row8693_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8693_block_006
  · decide
  · rfl

def row8693_block_007 : List Nat := [13619]
def plan8693_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.firstFactor 29), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row8693_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8693 row8693_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8693_block_007
  · decide
  · rfl

def row8693_block_008 : List Nat := [14783, 15581]
def plan8693_block_008 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 11), .first, .second, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 29), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 31)]
set_option maxRecDepth 2048 in
theorem row8693_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8693 row8693_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8693_block_008
  · decide
  · rfl

def row8693_block_009 : List Nat := [17291]
def plan8693_block_009 : List Budget.Choice := [.first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 29), (.secondFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.secondFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8693_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8693 row8693_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8693_block_009
  · decide
  · rfl

def row8693_block_010 : List Nat := []
theorem row8693_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8693 row8693_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8693_block_011 : List Nat := []
theorem row8693_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8693 row8693_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8693_block_012 : List Nat := []
theorem row8693_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8693 row8693_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8693_block_013 : List Nat := []
theorem row8693_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8693 row8693_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8693_large_left_left : List Nat := row8693_small ++ row8693_block_005
theorem row8693_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8693 row8693_large_left_left :=
  Budget.RowUpperBound.branch row8693_small_covers row8693_block_005_covers

def row8693_large_left_right_right : List Nat := row8693_block_007 ++ row8693_block_008
theorem row8693_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8693 row8693_large_left_right_right :=
  Budget.RowUpperBound.branch row8693_block_007_covers row8693_block_008_covers

def row8693_large_left_right : List Nat := row8693_block_006 ++ row8693_large_left_right_right
theorem row8693_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8693 row8693_large_left_right :=
  Budget.RowUpperBound.branch row8693_block_006_covers row8693_large_left_right_right_covers

def row8693_large_left : List Nat := row8693_large_left_left ++ row8693_large_left_right
theorem row8693_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8693 row8693_large_left :=
  Budget.RowUpperBound.branch row8693_large_left_left_covers row8693_large_left_right_covers

def row8693_large_right_left : List Nat := row8693_block_009 ++ row8693_block_010
theorem row8693_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8693 row8693_large_right_left :=
  Budget.RowUpperBound.branch row8693_block_009_covers row8693_block_010_covers

def row8693_large_right_right_right : List Nat := row8693_block_012 ++ row8693_block_013
theorem row8693_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8693 row8693_large_right_right_right :=
  Budget.RowUpperBound.branch row8693_block_012_covers row8693_block_013_covers

def row8693_large_right_right : List Nat := row8693_block_011 ++ row8693_large_right_right_right
theorem row8693_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8693 row8693_large_right_right :=
  Budget.RowUpperBound.branch row8693_block_011_covers row8693_large_right_right_right_covers

def row8693_large_right : List Nat := row8693_large_right_left ++ row8693_large_right_right
theorem row8693_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8693 row8693_large_right :=
  Budget.RowUpperBound.branch row8693_large_right_left_covers row8693_large_right_right_covers

def row8693_large : List Nat := row8693_large_left ++ row8693_large_right
theorem row8693_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8693 row8693_large :=
  Budget.RowUpperBound.branch row8693_large_left_covers row8693_large_right_covers

def row8693 : List Nat := row8693_large
theorem row8693_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8693 row8693 := row8693_large_covers

def row8699_small : List Nat := []
theorem row8699_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8699 row8699_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8699_block_005 : List Nat := [8963, 9281, 9323, 9467, 9929, 10103]
def plan8699_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .possible, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .second, .second, (.secondFactor 29), .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 19), .second, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8699_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8699 row8699_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8699_block_005
  · decide
  · rfl

def row8699_block_006 : List Nat := [10667, 11213, 11807, 11909]
def plan8699_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 19), .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .possible, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 11), .possible, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .second]
set_option maxRecDepth 2048 in
theorem row8699_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8699 row8699_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8699_block_006
  · decide
  · rfl

def row8699_block_007 : List Nat := [12503, 12983, 13007, 13313, 13883]
def plan8699_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 11), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row8699_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8699 row8699_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8699_block_007
  · decide
  · rfl

def row8699_block_008 : List Nat := [14699, 15461]
def plan8699_block_008 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), (.secondFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.secondFactor 11), .first, (.firstFactor 11), .second, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .second, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 19), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8699_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8699 row8699_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8699_block_008
  · decide
  · rfl

def row8699_block_009 : List Nat := [16829, 17117]
def plan8699_block_009 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 7), (.firstFactor 31), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .second, (.secondFactor 17), .first, .possible, (.secondFactor 31), .first, (.secondFactor 23), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8699_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8699 row8699_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8699_block_009
  · decide
  · rfl

def row8699_block_010 : List Nat := []
theorem row8699_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8699 row8699_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8699_block_011 : List Nat := []
theorem row8699_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8699 row8699_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8699_block_012 : List Nat := []
theorem row8699_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8699 row8699_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8699_block_013 : List Nat := []
theorem row8699_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8699 row8699_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8699_large_left_left : List Nat := row8699_small ++ row8699_block_005
theorem row8699_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8699 row8699_large_left_left :=
  Budget.RowUpperBound.branch row8699_small_covers row8699_block_005_covers

def row8699_large_left_right_right : List Nat := row8699_block_007 ++ row8699_block_008
theorem row8699_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8699 row8699_large_left_right_right :=
  Budget.RowUpperBound.branch row8699_block_007_covers row8699_block_008_covers

def row8699_large_left_right : List Nat := row8699_block_006 ++ row8699_large_left_right_right
theorem row8699_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8699 row8699_large_left_right :=
  Budget.RowUpperBound.branch row8699_block_006_covers row8699_large_left_right_right_covers

def row8699_large_left : List Nat := row8699_large_left_left ++ row8699_large_left_right
theorem row8699_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8699 row8699_large_left :=
  Budget.RowUpperBound.branch row8699_large_left_left_covers row8699_large_left_right_covers

def row8699_large_right_left : List Nat := row8699_block_009 ++ row8699_block_010
theorem row8699_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8699 row8699_large_right_left :=
  Budget.RowUpperBound.branch row8699_block_009_covers row8699_block_010_covers

def row8699_large_right_right_right : List Nat := row8699_block_012 ++ row8699_block_013
theorem row8699_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8699 row8699_large_right_right_right :=
  Budget.RowUpperBound.branch row8699_block_012_covers row8699_block_013_covers

def row8699_large_right_right : List Nat := row8699_block_011 ++ row8699_large_right_right_right
theorem row8699_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8699 row8699_large_right_right :=
  Budget.RowUpperBound.branch row8699_block_011_covers row8699_large_right_right_right_covers

def row8699_large_right : List Nat := row8699_large_right_left ++ row8699_large_right_right
theorem row8699_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8699 row8699_large_right :=
  Budget.RowUpperBound.branch row8699_large_right_left_covers row8699_large_right_right_covers

def row8699_large : List Nat := row8699_large_left ++ row8699_large_right
theorem row8699_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8699 row8699_large :=
  Budget.RowUpperBound.branch row8699_large_left_covers row8699_large_right_covers

def row8699 : List Nat := row8699_large
theorem row8699_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8699 row8699 := row8699_large_covers

def row8707_small : List Nat := []
theorem row8707_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8707 row8707_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8707_block_005 : List Nat := [8839, 9397, 10177]
def plan8707_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .possible, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.firstFactor 13), .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8707_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8707 row8707_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8707_block_005
  · decide
  · rfl

def row8707_block_006 : List Nat := [10723, 10837, 11131, 11239, 11971, 12253]
def plan8707_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .possible, (.firstFactor 7), .first, (.firstFactor 23), .first, .second, .first, .second, .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 31), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .second, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8707_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8707 row8707_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8707_block_006
  · decide
  · rfl

def row8707_block_007 : List Nat := [12421, 12967, 13033, 13297, 13537, 13873]
def plan8707_block_007 : List Budget.Choice := [.possible, .second, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 31), (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .possible, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 23), .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .second, .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .second]
set_option maxRecDepth 2048 in
theorem row8707_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8707 row8707_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8707_block_007
  · decide
  · rfl

def row8707_block_008 : List Nat := [15667]
def plan8707_block_008 : List Budget.Choice := [(.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .second, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), (.secondFactor 31), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8707_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8707 row8707_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8707_block_008
  · decide
  · rfl

def row8707_block_009 : List Nat := [16633, 16981]
def plan8707_block_009 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), .second, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 19), (.firstFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8707_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8707 row8707_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8707_block_009
  · decide
  · rfl

def row8707_block_010 : List Nat := []
theorem row8707_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8707 row8707_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8707_block_011 : List Nat := []
theorem row8707_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8707 row8707_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8707_block_012 : List Nat := []
theorem row8707_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8707 row8707_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8707_block_013 : List Nat := []
theorem row8707_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8707 row8707_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8707_large_left_left : List Nat := row8707_small ++ row8707_block_005
theorem row8707_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8707 row8707_large_left_left :=
  Budget.RowUpperBound.branch row8707_small_covers row8707_block_005_covers

def row8707_large_left_right_right : List Nat := row8707_block_007 ++ row8707_block_008
theorem row8707_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8707 row8707_large_left_right_right :=
  Budget.RowUpperBound.branch row8707_block_007_covers row8707_block_008_covers

def row8707_large_left_right : List Nat := row8707_block_006 ++ row8707_large_left_right_right
theorem row8707_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8707 row8707_large_left_right :=
  Budget.RowUpperBound.branch row8707_block_006_covers row8707_large_left_right_right_covers

def row8707_large_left : List Nat := row8707_large_left_left ++ row8707_large_left_right
theorem row8707_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8707 row8707_large_left :=
  Budget.RowUpperBound.branch row8707_large_left_left_covers row8707_large_left_right_covers

def row8707_large_right_left : List Nat := row8707_block_009 ++ row8707_block_010
theorem row8707_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8707 row8707_large_right_left :=
  Budget.RowUpperBound.branch row8707_block_009_covers row8707_block_010_covers

def row8707_large_right_right_right : List Nat := row8707_block_012 ++ row8707_block_013
theorem row8707_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8707 row8707_large_right_right_right :=
  Budget.RowUpperBound.branch row8707_block_012_covers row8707_block_013_covers

def row8707_large_right_right : List Nat := row8707_block_011 ++ row8707_large_right_right_right
theorem row8707_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8707 row8707_large_right_right :=
  Budget.RowUpperBound.branch row8707_block_011_covers row8707_large_right_right_right_covers

def row8707_large_right : List Nat := row8707_large_right_left ++ row8707_large_right_right
theorem row8707_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8707 row8707_large_right :=
  Budget.RowUpperBound.branch row8707_large_right_left_covers row8707_large_right_right_covers

def row8707_large : List Nat := row8707_large_left ++ row8707_large_right
theorem row8707_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8707 row8707_large :=
  Budget.RowUpperBound.branch row8707_large_left_covers row8707_large_right_covers

def row8707 : List Nat := row8707_large
theorem row8707_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8707 row8707 := row8707_large_covers

def row8713_small : List Nat := []
theorem row8713_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8713 row8713_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8713_block_005 : List Nat := [9319, 9439, 9859]
def plan8713_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 37), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 17), .first, .possible, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .second, .second, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 29), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8713_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8713 row8713_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8713_block_005
  · decide
  · rfl

def row8713_block_006 : List Nat := [10753, 11719, 11971]
def plan8713_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .possible, (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 31), (.secondFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, .second, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8713_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8713 row8713_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8713_block_006
  · decide
  · rfl

def row8713_block_007 : List Nat := [12601, 13873, 13879, 14071, 14449]
def plan8713_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .possible, .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .possible, .first, .possible, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .possible, .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first]
set_option maxRecDepth 2048 in
theorem row8713_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8713 row8713_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8713_block_007
  · decide
  · rfl

def row8713_block_008 : List Nat := [14713, 15241, 15349]
def plan8713_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .possible, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8713_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8713 row8713_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8713_block_008
  · decide
  · rfl

def row8713_block_009 : List Nat := [16603, 16921, 17011]
def plan8713_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 19), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, .second, .possible, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 23), (.secondFactor 7), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8713_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8713 row8713_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8713_block_009
  · decide
  · rfl

def row8713_block_010 : List Nat := []
theorem row8713_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8713 row8713_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8713_block_011 : List Nat := []
theorem row8713_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8713 row8713_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8713_block_012 : List Nat := []
theorem row8713_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8713 row8713_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8713_block_013 : List Nat := []
theorem row8713_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8713 row8713_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8713_large_left_left : List Nat := row8713_small ++ row8713_block_005
theorem row8713_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8713 row8713_large_left_left :=
  Budget.RowUpperBound.branch row8713_small_covers row8713_block_005_covers

def row8713_large_left_right_right : List Nat := row8713_block_007 ++ row8713_block_008
theorem row8713_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8713 row8713_large_left_right_right :=
  Budget.RowUpperBound.branch row8713_block_007_covers row8713_block_008_covers

def row8713_large_left_right : List Nat := row8713_block_006 ++ row8713_large_left_right_right
theorem row8713_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8713 row8713_large_left_right :=
  Budget.RowUpperBound.branch row8713_block_006_covers row8713_large_left_right_right_covers

def row8713_large_left : List Nat := row8713_large_left_left ++ row8713_large_left_right
theorem row8713_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8713 row8713_large_left :=
  Budget.RowUpperBound.branch row8713_large_left_left_covers row8713_large_left_right_covers

def row8713_large_right_left : List Nat := row8713_block_009 ++ row8713_block_010
theorem row8713_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8713 row8713_large_right_left :=
  Budget.RowUpperBound.branch row8713_block_009_covers row8713_block_010_covers

def row8713_large_right_right_right : List Nat := row8713_block_012 ++ row8713_block_013
theorem row8713_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8713 row8713_large_right_right_right :=
  Budget.RowUpperBound.branch row8713_block_012_covers row8713_block_013_covers

def row8713_large_right_right : List Nat := row8713_block_011 ++ row8713_large_right_right_right
theorem row8713_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8713 row8713_large_right_right :=
  Budget.RowUpperBound.branch row8713_block_011_covers row8713_large_right_right_right_covers

def row8713_large_right : List Nat := row8713_large_right_left ++ row8713_large_right_right
theorem row8713_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8713 row8713_large_right :=
  Budget.RowUpperBound.branch row8713_large_right_left_covers row8713_large_right_right_covers

def row8713_large : List Nat := row8713_large_left ++ row8713_large_right
theorem row8713_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8713 row8713_large :=
  Budget.RowUpperBound.branch row8713_large_left_covers row8713_large_right_covers

def row8713 : List Nat := row8713_large
theorem row8713_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8713 row8713 := row8713_large_covers

end ProjectEulerSolutions.PrimePairLarge
