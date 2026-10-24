import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10181_small : List Nat := []
theorem row10181_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10181 row10181_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10181_block_005 : List Nat := [10271, 10343]
def plan10181_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row10181_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10181 row10181_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10181_block_005
  · decide
  · rfl

def row10181_block_006 : List Nat := [11279, 11519, 11867, 12281]
def plan10181_block_006 : List Budget.Choice := [.second, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.secondFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 29), (.firstFactor 29), .first, .first, (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .second, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .second, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 37), (.secondFactor 23), (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, .second, .first, .second]
set_option maxRecDepth 2048 in
theorem row10181_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10181 row10181_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10181_block_006
  · decide
  · rfl

def row10181_block_007 : List Nat := [14057, 14177]
def plan10181_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 29), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 31), .second, (.secondFactor 29), (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 23), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .second, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10181_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10181 row10181_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10181_block_007
  · decide
  · rfl

def row10181_block_008 : List Nat := [15053, 15473]
def plan10181_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .possible, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 11), (.secondFactor 19), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10181_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10181 row10181_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10181_block_008
  · decide
  · rfl

def row10181_block_009 : List Nat := []
theorem row10181_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10181 row10181_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10181_block_010 : List Nat := []
theorem row10181_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10181 row10181_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10181_block_011 : List Nat := []
theorem row10181_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10181 row10181_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10181_block_012 : List Nat := []
theorem row10181_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10181 row10181_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10181_block_013 : List Nat := []
theorem row10181_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10181 row10181_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10181_large_left_left : List Nat := row10181_small ++ row10181_block_005
theorem row10181_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10181 row10181_large_left_left :=
  Budget.RowUpperBound.branch row10181_small_covers row10181_block_005_covers

def row10181_large_left_right_right : List Nat := row10181_block_007 ++ row10181_block_008
theorem row10181_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10181 row10181_large_left_right_right :=
  Budget.RowUpperBound.branch row10181_block_007_covers row10181_block_008_covers

def row10181_large_left_right : List Nat := row10181_block_006 ++ row10181_large_left_right_right
theorem row10181_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10181 row10181_large_left_right :=
  Budget.RowUpperBound.branch row10181_block_006_covers row10181_large_left_right_right_covers

def row10181_large_left : List Nat := row10181_large_left_left ++ row10181_large_left_right
theorem row10181_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10181 row10181_large_left :=
  Budget.RowUpperBound.branch row10181_large_left_left_covers row10181_large_left_right_covers

def row10181_large_right_left : List Nat := row10181_block_009 ++ row10181_block_010
theorem row10181_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10181 row10181_large_right_left :=
  Budget.RowUpperBound.branch row10181_block_009_covers row10181_block_010_covers

def row10181_large_right_right_right : List Nat := row10181_block_012 ++ row10181_block_013
theorem row10181_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10181 row10181_large_right_right_right :=
  Budget.RowUpperBound.branch row10181_block_012_covers row10181_block_013_covers

def row10181_large_right_right : List Nat := row10181_block_011 ++ row10181_large_right_right_right
theorem row10181_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10181 row10181_large_right_right :=
  Budget.RowUpperBound.branch row10181_block_011_covers row10181_large_right_right_right_covers

def row10181_large_right : List Nat := row10181_large_right_left ++ row10181_large_right_right
theorem row10181_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10181 row10181_large_right :=
  Budget.RowUpperBound.branch row10181_large_right_left_covers row10181_large_right_right_covers

def row10181_large : List Nat := row10181_large_left ++ row10181_large_right
theorem row10181_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10181 row10181_large :=
  Budget.RowUpperBound.branch row10181_large_left_covers row10181_large_right_covers

def row10181 : List Nat := row10181_large
theorem row10181_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10181 row10181 := row10181_large_covers

def row10193_small : List Nat := []
theorem row10193_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10193 row10193_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10193_block_005 : List Nat := []
def plan10193_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10193_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10193 row10193_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10193_block_005
  · decide
  · rfl

def row10193_block_006 : List Nat := [10529, 12263]
def plan10193_block_006 : List Budget.Choice := [(.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 7), (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 37), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 37), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, .first, .possible, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row10193_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10193 row10193_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10193_block_006
  · decide
  · rfl

def row10193_block_007 : List Nat := [13469, 13499]
def plan10193_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), (.secondFactor 37), (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, (.secondFactor 19), .possible, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10193_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10193 row10193_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10193_block_007
  · decide
  · rfl

def row10193_block_008 : List Nat := []
def plan10193_block_008 : List Budget.Choice := [.first, (.firstFactor 19), .first, .second, .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .second, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .second, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), (.firstFactor 29), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10193_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10193 row10193_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10193_block_008
  · decide
  · rfl

def row10193_block_009 : List Nat := []
theorem row10193_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10193 row10193_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10193_block_010 : List Nat := []
theorem row10193_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10193 row10193_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10193_block_011 : List Nat := []
theorem row10193_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10193 row10193_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10193_block_012 : List Nat := []
theorem row10193_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10193 row10193_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10193_block_013 : List Nat := []
theorem row10193_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10193 row10193_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10193_large_left_left : List Nat := row10193_small ++ row10193_block_005
theorem row10193_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10193 row10193_large_left_left :=
  Budget.RowUpperBound.branch row10193_small_covers row10193_block_005_covers

def row10193_large_left_right_right : List Nat := row10193_block_007 ++ row10193_block_008
theorem row10193_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10193 row10193_large_left_right_right :=
  Budget.RowUpperBound.branch row10193_block_007_covers row10193_block_008_covers

def row10193_large_left_right : List Nat := row10193_block_006 ++ row10193_large_left_right_right
theorem row10193_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10193 row10193_large_left_right :=
  Budget.RowUpperBound.branch row10193_block_006_covers row10193_large_left_right_right_covers

def row10193_large_left : List Nat := row10193_large_left_left ++ row10193_large_left_right
theorem row10193_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10193 row10193_large_left :=
  Budget.RowUpperBound.branch row10193_large_left_left_covers row10193_large_left_right_covers

def row10193_large_right_left : List Nat := row10193_block_009 ++ row10193_block_010
theorem row10193_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10193 row10193_large_right_left :=
  Budget.RowUpperBound.branch row10193_block_009_covers row10193_block_010_covers

def row10193_large_right_right_right : List Nat := row10193_block_012 ++ row10193_block_013
theorem row10193_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10193 row10193_large_right_right_right :=
  Budget.RowUpperBound.branch row10193_block_012_covers row10193_block_013_covers

def row10193_large_right_right : List Nat := row10193_block_011 ++ row10193_large_right_right_right
theorem row10193_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10193 row10193_large_right_right :=
  Budget.RowUpperBound.branch row10193_block_011_covers row10193_large_right_right_right_covers

def row10193_large_right : List Nat := row10193_large_right_left ++ row10193_large_right_right
theorem row10193_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10193 row10193_large_right :=
  Budget.RowUpperBound.branch row10193_large_right_left_covers row10193_large_right_right_covers

def row10193_large : List Nat := row10193_large_left ++ row10193_large_right
theorem row10193_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10193 row10193_large :=
  Budget.RowUpperBound.branch row10193_large_left_covers row10193_large_right_covers

def row10193 : List Nat := row10193_large
theorem row10193_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10193 row10193 := row10193_large_covers

def row10211_small : List Nat := []
theorem row10211_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10211 row10211_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10211_block_005 : List Nat := []
def plan10211_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row10211_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10211 row10211_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10211_block_005
  · decide
  · rfl

def row10211_block_006 : List Nat := [10589]
def plan10211_block_006 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 37), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, .first, .second]
set_option maxRecDepth 2048 in
theorem row10211_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10211 row10211_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10211_block_006
  · decide
  · rfl

def row10211_block_007 : List Nat := [13883, 14411]
def plan10211_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .second, (.firstFactor 31), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 37), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .second, .first, .second, .second, .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.secondFactor 17), .first, .first, (.secondFactor 37), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row10211_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10211 row10211_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10211_block_007
  · decide
  · rfl

def row10211_block_008 : List Nat := [15101, 15737]
def plan10211_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 17), (.secondFactor 13), .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10211_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10211 row10211_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10211_block_008
  · decide
  · rfl

def row10211_block_009 : List Nat := []
theorem row10211_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10211 row10211_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10211_block_010 : List Nat := []
theorem row10211_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10211 row10211_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10211_block_011 : List Nat := []
theorem row10211_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10211 row10211_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10211_block_012 : List Nat := []
theorem row10211_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10211 row10211_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10211_block_013 : List Nat := []
theorem row10211_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10211 row10211_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10211_large_left_left : List Nat := row10211_small ++ row10211_block_005
theorem row10211_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10211 row10211_large_left_left :=
  Budget.RowUpperBound.branch row10211_small_covers row10211_block_005_covers

def row10211_large_left_right_right : List Nat := row10211_block_007 ++ row10211_block_008
theorem row10211_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10211 row10211_large_left_right_right :=
  Budget.RowUpperBound.branch row10211_block_007_covers row10211_block_008_covers

def row10211_large_left_right : List Nat := row10211_block_006 ++ row10211_large_left_right_right
theorem row10211_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10211 row10211_large_left_right :=
  Budget.RowUpperBound.branch row10211_block_006_covers row10211_large_left_right_right_covers

def row10211_large_left : List Nat := row10211_large_left_left ++ row10211_large_left_right
theorem row10211_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10211 row10211_large_left :=
  Budget.RowUpperBound.branch row10211_large_left_left_covers row10211_large_left_right_covers

def row10211_large_right_left : List Nat := row10211_block_009 ++ row10211_block_010
theorem row10211_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10211 row10211_large_right_left :=
  Budget.RowUpperBound.branch row10211_block_009_covers row10211_block_010_covers

def row10211_large_right_right_right : List Nat := row10211_block_012 ++ row10211_block_013
theorem row10211_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10211 row10211_large_right_right_right :=
  Budget.RowUpperBound.branch row10211_block_012_covers row10211_block_013_covers

def row10211_large_right_right : List Nat := row10211_block_011 ++ row10211_large_right_right_right
theorem row10211_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10211 row10211_large_right_right :=
  Budget.RowUpperBound.branch row10211_block_011_covers row10211_large_right_right_right_covers

def row10211_large_right : List Nat := row10211_large_right_left ++ row10211_large_right_right
theorem row10211_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10211 row10211_large_right :=
  Budget.RowUpperBound.branch row10211_large_right_left_covers row10211_large_right_right_covers

def row10211_large : List Nat := row10211_large_left ++ row10211_large_right
theorem row10211_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10211 row10211_large :=
  Budget.RowUpperBound.branch row10211_large_left_covers row10211_large_right_covers

def row10211 : List Nat := row10211_large
theorem row10211_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10211 row10211 := row10211_large_covers

def row10223_small : List Nat := []
theorem row10223_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10223 row10223_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10223_block_005 : List Nat := []
def plan10223_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 31), .first, .first]
set_option maxRecDepth 2048 in
theorem row10223_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10223 row10223_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10223_block_005
  · decide
  · rfl

def row10223_block_006 : List Nat := [10733, 10847, 10859, 11447, 11789, 11987, 12107, 12113]
def plan10223_block_006 : List Budget.Choice := [(.secondFactor 29), .first, .second, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 17), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 23), .possible, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .possible, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .possible, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .possible, .first, .possible, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row10223_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10223 row10223_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10223_block_006
  · decide
  · rfl

def row10223_block_007 : List Nat := [12539, 12821, 13163, 13217]
def plan10223_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .possible, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 19), .first, (.firstFactor 37), (.firstFactor 13), (.secondFactor 17), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.secondFactor 37), .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 23), .first, .second, .first, .first, .first, .second, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10223_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10223 row10223_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10223_block_007
  · decide
  · rfl

def row10223_block_008 : List Nat := [14519, 14639, 14897, 15527]
def plan10223_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.secondFactor 31), (.secondFactor 13), .first, .first, .first, .first, .second, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 31), (.firstFactor 11), .first, (.secondFactor 23), .second, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 37), (.secondFactor 31), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .second, (.firstFactor 37), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10223_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10223 row10223_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10223_block_008
  · decide
  · rfl

def row10223_block_009 : List Nat := []
theorem row10223_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10223 row10223_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10223_block_010 : List Nat := []
theorem row10223_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10223 row10223_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10223_block_011 : List Nat := []
theorem row10223_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10223 row10223_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10223_block_012 : List Nat := []
theorem row10223_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10223 row10223_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10223_block_013 : List Nat := []
theorem row10223_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10223 row10223_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10223_large_left_left : List Nat := row10223_small ++ row10223_block_005
theorem row10223_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10223 row10223_large_left_left :=
  Budget.RowUpperBound.branch row10223_small_covers row10223_block_005_covers

def row10223_large_left_right_right : List Nat := row10223_block_007 ++ row10223_block_008
theorem row10223_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10223 row10223_large_left_right_right :=
  Budget.RowUpperBound.branch row10223_block_007_covers row10223_block_008_covers

def row10223_large_left_right : List Nat := row10223_block_006 ++ row10223_large_left_right_right
theorem row10223_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10223 row10223_large_left_right :=
  Budget.RowUpperBound.branch row10223_block_006_covers row10223_large_left_right_right_covers

def row10223_large_left : List Nat := row10223_large_left_left ++ row10223_large_left_right
theorem row10223_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10223 row10223_large_left :=
  Budget.RowUpperBound.branch row10223_large_left_left_covers row10223_large_left_right_covers

def row10223_large_right_left : List Nat := row10223_block_009 ++ row10223_block_010
theorem row10223_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10223 row10223_large_right_left :=
  Budget.RowUpperBound.branch row10223_block_009_covers row10223_block_010_covers

def row10223_large_right_right_right : List Nat := row10223_block_012 ++ row10223_block_013
theorem row10223_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10223 row10223_large_right_right_right :=
  Budget.RowUpperBound.branch row10223_block_012_covers row10223_block_013_covers

def row10223_large_right_right : List Nat := row10223_block_011 ++ row10223_large_right_right_right
theorem row10223_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10223 row10223_large_right_right :=
  Budget.RowUpperBound.branch row10223_block_011_covers row10223_large_right_right_right_covers

def row10223_large_right : List Nat := row10223_large_right_left ++ row10223_large_right_right
theorem row10223_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10223 row10223_large_right :=
  Budget.RowUpperBound.branch row10223_large_right_left_covers row10223_large_right_right_covers

def row10223_large : List Nat := row10223_large_left ++ row10223_large_right
theorem row10223_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10223 row10223_large :=
  Budget.RowUpperBound.branch row10223_large_left_covers row10223_large_right_covers

def row10223 : List Nat := row10223_large
theorem row10223_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10223 row10223 := row10223_large_covers

end ProjectEulerSolutions.PrimePairLarge
