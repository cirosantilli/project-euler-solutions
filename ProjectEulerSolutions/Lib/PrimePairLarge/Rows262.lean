import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8389_small : List Nat := []
theorem row8389_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8389 row8389_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8389_block_005 : List Nat := [8677, 8887, 8929, 9007, 9133, 9643]
def plan8389_block_005 : List Budget.Choice := [.outside, (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 23), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 13), (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), .possible, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 31), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 19), (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.secondFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 19), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8389_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8389 row8389_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8389_block_005
  · decide
  · rfl

def row8389_block_006 : List Nat := [10903, 10939, 12007]
def plan8389_block_006 : List Budget.Choice := [.first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 7), .possible, .second, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 23), .second, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .second, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .second, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .possible, .first, (.secondFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8389_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8389 row8389_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8389_block_006
  · decide
  · rfl

def row8389_block_007 : List Nat := [13219, 13687, 14149]
def plan8389_block_007 : List Budget.Choice := [(.firstFactor 31), (.firstFactor 37), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 11), (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .possible, .first, .first, .second, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 31), (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .second, .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8389_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8389 row8389_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8389_block_007
  · decide
  · rfl

def row8389_block_008 : List Nat := [15013, 15619]
def plan8389_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .possible, .first, (.firstFactor 17), .first, .second, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), .second, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 19), .possible, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .second, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8389_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8389 row8389_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8389_block_008
  · decide
  · rfl

def row8389_block_009 : List Nat := [17623]
def plan8389_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .possible, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8389_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8389 row8389_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8389_block_009
  · decide
  · rfl

def row8389_block_010 : List Nat := []
theorem row8389_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8389 row8389_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8389_block_011 : List Nat := []
theorem row8389_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8389 row8389_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8389_block_012 : List Nat := []
theorem row8389_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8389 row8389_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8389_block_013 : List Nat := []
theorem row8389_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8389 row8389_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8389_large_left_left : List Nat := row8389_small ++ row8389_block_005
theorem row8389_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8389 row8389_large_left_left :=
  Budget.RowUpperBound.branch row8389_small_covers row8389_block_005_covers

def row8389_large_left_right_right : List Nat := row8389_block_007 ++ row8389_block_008
theorem row8389_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8389 row8389_large_left_right_right :=
  Budget.RowUpperBound.branch row8389_block_007_covers row8389_block_008_covers

def row8389_large_left_right : List Nat := row8389_block_006 ++ row8389_large_left_right_right
theorem row8389_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8389 row8389_large_left_right :=
  Budget.RowUpperBound.branch row8389_block_006_covers row8389_large_left_right_right_covers

def row8389_large_left : List Nat := row8389_large_left_left ++ row8389_large_left_right
theorem row8389_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8389 row8389_large_left :=
  Budget.RowUpperBound.branch row8389_large_left_left_covers row8389_large_left_right_covers

def row8389_large_right_left : List Nat := row8389_block_009 ++ row8389_block_010
theorem row8389_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8389 row8389_large_right_left :=
  Budget.RowUpperBound.branch row8389_block_009_covers row8389_block_010_covers

def row8389_large_right_right_right : List Nat := row8389_block_012 ++ row8389_block_013
theorem row8389_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8389 row8389_large_right_right_right :=
  Budget.RowUpperBound.branch row8389_block_012_covers row8389_block_013_covers

def row8389_large_right_right : List Nat := row8389_block_011 ++ row8389_large_right_right_right
theorem row8389_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8389 row8389_large_right_right :=
  Budget.RowUpperBound.branch row8389_block_011_covers row8389_large_right_right_right_covers

def row8389_large_right : List Nat := row8389_large_right_left ++ row8389_large_right_right
theorem row8389_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8389 row8389_large_right :=
  Budget.RowUpperBound.branch row8389_large_right_left_covers row8389_large_right_right_covers

def row8389_large : List Nat := row8389_large_left ++ row8389_large_right
theorem row8389_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8389 row8389_large :=
  Budget.RowUpperBound.branch row8389_large_left_covers row8389_large_right_covers

def row8389 : List Nat := row8389_large
theorem row8389_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8389 row8389 := row8389_large_covers

def row8419_small : List Nat := []
theorem row8419_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8419 row8419_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8419_block_005 : List Nat := [8641, 8923, 9277, 9319, 9649, 9883, 9949]
def plan8419_block_005 : List Budget.Choice := [.outside, .outside, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 29), .possible, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .second, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .second, (.secondFactor 31), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .possible, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row8419_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8419 row8419_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8419_block_005
  · decide
  · rfl

def row8419_block_006 : List Nat := [10711, 10789, 11503]
def plan8419_block_006 : List Budget.Choice := [.first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 29), .first, .first, .possible, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row8419_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8419 row8419_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8419_block_006
  · decide
  · rfl

def row8419_block_007 : List Nat := []
def plan8419_block_007 : List Budget.Choice := [(.firstFactor 7), .second, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .second, .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 31)]
set_option maxRecDepth 2048 in
theorem row8419_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8419 row8419_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8419_block_007
  · decide
  · rfl

def row8419_block_008 : List Nat := [15937]
def plan8419_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .possible, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8419_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8419 row8419_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8419_block_008
  · decide
  · rfl

def row8419_block_009 : List Nat := [17317, 17443]
def plan8419_block_009 : List Budget.Choice := [.first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 31), (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .possible, .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8419_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8419 row8419_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8419_block_009
  · decide
  · rfl

def row8419_block_010 : List Nat := []
theorem row8419_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8419 row8419_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8419_block_011 : List Nat := []
theorem row8419_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8419 row8419_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8419_block_012 : List Nat := []
theorem row8419_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8419 row8419_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8419_block_013 : List Nat := []
theorem row8419_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8419 row8419_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8419_large_left_left : List Nat := row8419_small ++ row8419_block_005
theorem row8419_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8419 row8419_large_left_left :=
  Budget.RowUpperBound.branch row8419_small_covers row8419_block_005_covers

def row8419_large_left_right_right : List Nat := row8419_block_007 ++ row8419_block_008
theorem row8419_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8419 row8419_large_left_right_right :=
  Budget.RowUpperBound.branch row8419_block_007_covers row8419_block_008_covers

def row8419_large_left_right : List Nat := row8419_block_006 ++ row8419_large_left_right_right
theorem row8419_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8419 row8419_large_left_right :=
  Budget.RowUpperBound.branch row8419_block_006_covers row8419_large_left_right_right_covers

def row8419_large_left : List Nat := row8419_large_left_left ++ row8419_large_left_right
theorem row8419_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8419 row8419_large_left :=
  Budget.RowUpperBound.branch row8419_large_left_left_covers row8419_large_left_right_covers

def row8419_large_right_left : List Nat := row8419_block_009 ++ row8419_block_010
theorem row8419_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8419 row8419_large_right_left :=
  Budget.RowUpperBound.branch row8419_block_009_covers row8419_block_010_covers

def row8419_large_right_right_right : List Nat := row8419_block_012 ++ row8419_block_013
theorem row8419_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8419 row8419_large_right_right_right :=
  Budget.RowUpperBound.branch row8419_block_012_covers row8419_block_013_covers

def row8419_large_right_right : List Nat := row8419_block_011 ++ row8419_large_right_right_right
theorem row8419_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8419 row8419_large_right_right :=
  Budget.RowUpperBound.branch row8419_block_011_covers row8419_large_right_right_right_covers

def row8419_large_right : List Nat := row8419_large_right_left ++ row8419_large_right_right
theorem row8419_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8419 row8419_large_right :=
  Budget.RowUpperBound.branch row8419_large_right_left_covers row8419_large_right_right_covers

def row8419_large : List Nat := row8419_large_left ++ row8419_large_right
theorem row8419_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8419 row8419_large :=
  Budget.RowUpperBound.branch row8419_large_left_covers row8419_large_right_covers

def row8419 : List Nat := row8419_large
theorem row8419_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8419 row8419 := row8419_large_covers

def row8423_small : List Nat := []
theorem row8423_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8423 row8423_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8423_block_005 : List Nat := [8513, 8573, 8699, 9137, 9377]
def plan8423_block_005 : List Budget.Choice := [.outside, .outside, .outside, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .possible, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .possible, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .possible, .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), .second, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .second, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .second, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8423_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8423 row8423_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8423_block_005
  · decide
  · rfl

def row8423_block_006 : List Nat := [11213, 12269]
def plan8423_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .second, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 29), .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .possible, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8423_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8423 row8423_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8423_block_006
  · decide
  · rfl

def row8423_block_007 : List Nat := [12791, 13151, 13397, 13463, 13487]
def plan8423_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .possible, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8423_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8423 row8423_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8423_block_007
  · decide
  · rfl

def row8423_block_008 : List Nat := [14639, 15629, 16253]
def plan8423_block_008 : List Budget.Choice := [.first, .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 23), (.secondFactor 29), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 23), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .second, .first, (.secondFactor 23), (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .second, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8423_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8423 row8423_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8423_block_008
  · decide
  · rfl

def row8423_block_009 : List Nat := []
def plan8423_block_009 : List Budget.Choice := [.first, (.secondFactor 19), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 37), (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8423_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8423 row8423_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8423_block_009
  · decide
  · rfl

def row8423_block_010 : List Nat := []
theorem row8423_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8423 row8423_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8423_block_011 : List Nat := []
theorem row8423_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8423 row8423_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8423_block_012 : List Nat := []
theorem row8423_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8423 row8423_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8423_block_013 : List Nat := []
theorem row8423_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8423 row8423_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8423_large_left_left : List Nat := row8423_small ++ row8423_block_005
theorem row8423_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8423 row8423_large_left_left :=
  Budget.RowUpperBound.branch row8423_small_covers row8423_block_005_covers

def row8423_large_left_right_right : List Nat := row8423_block_007 ++ row8423_block_008
theorem row8423_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8423 row8423_large_left_right_right :=
  Budget.RowUpperBound.branch row8423_block_007_covers row8423_block_008_covers

def row8423_large_left_right : List Nat := row8423_block_006 ++ row8423_large_left_right_right
theorem row8423_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8423 row8423_large_left_right :=
  Budget.RowUpperBound.branch row8423_block_006_covers row8423_large_left_right_right_covers

def row8423_large_left : List Nat := row8423_large_left_left ++ row8423_large_left_right
theorem row8423_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8423 row8423_large_left :=
  Budget.RowUpperBound.branch row8423_large_left_left_covers row8423_large_left_right_covers

def row8423_large_right_left : List Nat := row8423_block_009 ++ row8423_block_010
theorem row8423_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8423 row8423_large_right_left :=
  Budget.RowUpperBound.branch row8423_block_009_covers row8423_block_010_covers

def row8423_large_right_right_right : List Nat := row8423_block_012 ++ row8423_block_013
theorem row8423_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8423 row8423_large_right_right_right :=
  Budget.RowUpperBound.branch row8423_block_012_covers row8423_block_013_covers

def row8423_large_right_right : List Nat := row8423_block_011 ++ row8423_large_right_right_right
theorem row8423_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8423 row8423_large_right_right :=
  Budget.RowUpperBound.branch row8423_block_011_covers row8423_large_right_right_right_covers

def row8423_large_right : List Nat := row8423_large_right_left ++ row8423_large_right_right
theorem row8423_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8423 row8423_large_right :=
  Budget.RowUpperBound.branch row8423_large_right_left_covers row8423_large_right_right_covers

def row8423_large : List Nat := row8423_large_left ++ row8423_large_right
theorem row8423_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8423 row8423_large :=
  Budget.RowUpperBound.branch row8423_large_left_covers row8423_large_right_covers

def row8423 : List Nat := row8423_large
theorem row8423_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8423 row8423 := row8423_large_covers

def row8429_small : List Nat := []
theorem row8429_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8429 row8429_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8429_block_005 : List Nat := [8693, 8807, 8831, 10151]
def plan8429_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .possible, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 11), .first, .possible, (.firstFactor 7), .first, .second, (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .second, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .second, .first, (.firstFactor 7), .second, (.firstFactor 11), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .possible, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 31), .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row8429_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8429 row8429_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8429_block_005
  · decide
  · rfl

def row8429_block_006 : List Nat := [10457, 11411]
def plan8429_block_006 : List Budget.Choice := [(.secondFactor 19), .first, .second, .first, (.firstFactor 19), .first, .possible, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .second, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.secondFactor 11), .first, .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8429_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8429 row8429_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8429_block_006
  · decide
  · rfl

def row8429_block_007 : List Nat := [12821, 13121, 13457, 14447]
def plan8429_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .second, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8429_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8429 row8429_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8429_block_007
  · decide
  · rfl

def row8429_block_008 : List Nat := [14543, 15413, 15467, 15497, 15959]
def plan8429_block_008 : List Budget.Choice := [.first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .possible, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .possible, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 31), .second, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.secondFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row8429_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8429 row8429_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8429_block_008
  · decide
  · rfl

def row8429_block_009 : List Nat := [16901, 16979]
def plan8429_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 19), (.secondFactor 11), (.firstFactor 11), .first, .possible, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8429_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8429 row8429_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8429_block_009
  · decide
  · rfl

def row8429_block_010 : List Nat := []
theorem row8429_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8429 row8429_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8429_block_011 : List Nat := []
theorem row8429_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8429 row8429_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8429_block_012 : List Nat := []
theorem row8429_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8429 row8429_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8429_block_013 : List Nat := []
theorem row8429_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8429 row8429_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8429_large_left_left : List Nat := row8429_small ++ row8429_block_005
theorem row8429_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8429 row8429_large_left_left :=
  Budget.RowUpperBound.branch row8429_small_covers row8429_block_005_covers

def row8429_large_left_right_right : List Nat := row8429_block_007 ++ row8429_block_008
theorem row8429_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8429 row8429_large_left_right_right :=
  Budget.RowUpperBound.branch row8429_block_007_covers row8429_block_008_covers

def row8429_large_left_right : List Nat := row8429_block_006 ++ row8429_large_left_right_right
theorem row8429_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8429 row8429_large_left_right :=
  Budget.RowUpperBound.branch row8429_block_006_covers row8429_large_left_right_right_covers

def row8429_large_left : List Nat := row8429_large_left_left ++ row8429_large_left_right
theorem row8429_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8429 row8429_large_left :=
  Budget.RowUpperBound.branch row8429_large_left_left_covers row8429_large_left_right_covers

def row8429_large_right_left : List Nat := row8429_block_009 ++ row8429_block_010
theorem row8429_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8429 row8429_large_right_left :=
  Budget.RowUpperBound.branch row8429_block_009_covers row8429_block_010_covers

def row8429_large_right_right_right : List Nat := row8429_block_012 ++ row8429_block_013
theorem row8429_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8429 row8429_large_right_right_right :=
  Budget.RowUpperBound.branch row8429_block_012_covers row8429_block_013_covers

def row8429_large_right_right : List Nat := row8429_block_011 ++ row8429_large_right_right_right
theorem row8429_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8429 row8429_large_right_right :=
  Budget.RowUpperBound.branch row8429_block_011_covers row8429_large_right_right_right_covers

def row8429_large_right : List Nat := row8429_large_right_left ++ row8429_large_right_right
theorem row8429_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8429 row8429_large_right :=
  Budget.RowUpperBound.branch row8429_large_right_left_covers row8429_large_right_right_covers

def row8429_large : List Nat := row8429_large_left ++ row8429_large_right
theorem row8429_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8429 row8429_large :=
  Budget.RowUpperBound.branch row8429_large_left_covers row8429_large_right_covers

def row8429 : List Nat := row8429_large
theorem row8429_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8429 row8429 := row8429_large_covers

end ProjectEulerSolutions.PrimePairLarge
