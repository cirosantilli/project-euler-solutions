import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10243_small : List Nat := []
theorem row10243_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10243 row10243_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10243_block_005 : List Nat := [10303]
def plan10243_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row10243_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10243 row10243_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10243_block_005
  · decide
  · rfl

def row10243_block_006 : List Nat := [10477, 10663, 11311, 11731]
def plan10243_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .possible, .first, .second, .first, .first, (.firstFactor 29), (.secondFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 11), .possible, (.secondFactor 19), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10243_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10243 row10243_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10243_block_006
  · decide
  · rfl

def row10243_block_007 : List Nat := [13183, 13627, 14419]
def plan10243_block_007 : List Budget.Choice := [(.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 11), .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .second, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .second, (.secondFactor 13), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 37), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second]
set_option maxRecDepth 2048 in
theorem row10243_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10243 row10243_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10243_block_007
  · decide
  · rfl

def row10243_block_008 : List Nat := [15139, 15277, 15679]
def plan10243_block_008 : List Budget.Choice := [(.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .second, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10243_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10243 row10243_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10243_block_008
  · decide
  · rfl

def row10243_block_009 : List Nat := []
theorem row10243_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10243 row10243_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10243_block_010 : List Nat := []
theorem row10243_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10243 row10243_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10243_block_011 : List Nat := []
theorem row10243_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10243 row10243_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10243_block_012 : List Nat := []
theorem row10243_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10243 row10243_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10243_block_013 : List Nat := []
theorem row10243_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10243 row10243_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10243_large_left_left : List Nat := row10243_small ++ row10243_block_005
theorem row10243_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10243 row10243_large_left_left :=
  Budget.RowUpperBound.branch row10243_small_covers row10243_block_005_covers

def row10243_large_left_right_right : List Nat := row10243_block_007 ++ row10243_block_008
theorem row10243_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10243 row10243_large_left_right_right :=
  Budget.RowUpperBound.branch row10243_block_007_covers row10243_block_008_covers

def row10243_large_left_right : List Nat := row10243_block_006 ++ row10243_large_left_right_right
theorem row10243_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10243 row10243_large_left_right :=
  Budget.RowUpperBound.branch row10243_block_006_covers row10243_large_left_right_right_covers

def row10243_large_left : List Nat := row10243_large_left_left ++ row10243_large_left_right
theorem row10243_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10243 row10243_large_left :=
  Budget.RowUpperBound.branch row10243_large_left_left_covers row10243_large_left_right_covers

def row10243_large_right_left : List Nat := row10243_block_009 ++ row10243_block_010
theorem row10243_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10243 row10243_large_right_left :=
  Budget.RowUpperBound.branch row10243_block_009_covers row10243_block_010_covers

def row10243_large_right_right_right : List Nat := row10243_block_012 ++ row10243_block_013
theorem row10243_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10243 row10243_large_right_right_right :=
  Budget.RowUpperBound.branch row10243_block_012_covers row10243_block_013_covers

def row10243_large_right_right : List Nat := row10243_block_011 ++ row10243_large_right_right_right
theorem row10243_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10243 row10243_large_right_right :=
  Budget.RowUpperBound.branch row10243_block_011_covers row10243_large_right_right_right_covers

def row10243_large_right : List Nat := row10243_large_right_left ++ row10243_large_right_right
theorem row10243_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10243 row10243_large_right :=
  Budget.RowUpperBound.branch row10243_large_right_left_covers row10243_large_right_right_covers

def row10243_large : List Nat := row10243_large_left ++ row10243_large_right
theorem row10243_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10243 row10243_large :=
  Budget.RowUpperBound.branch row10243_large_left_covers row10243_large_right_covers

def row10243 : List Nat := row10243_large
theorem row10243_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10243 row10243 := row10243_large_covers

def row10247_small : List Nat := []
theorem row10247_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10247 row10247_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10247_block_005 : List Nat := []
def plan10247_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row10247_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10247 row10247_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10247_block_005
  · decide
  · rfl

def row10247_block_006 : List Nat := [10457, 11279]
def plan10247_block_006 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 17), (.firstFactor 31), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 17), .first, .second, .first, .first, .second, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 37), (.secondFactor 13), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row10247_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10247 row10247_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10247_block_006
  · decide
  · rfl

def row10247_block_007 : List Nat := [12527, 12641, 12917, 14057]
def plan10247_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 19), .first, .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 23), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 37), (.secondFactor 19), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .second, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), .first, .second, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), .possible, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10247_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10247 row10247_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10247_block_007
  · decide
  · rfl

def row10247_block_008 : List Nat := [15161, 15383]
def plan10247_block_008 : List Budget.Choice := [.first, .second, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .possible, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .second, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10247_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10247 row10247_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10247_block_008
  · decide
  · rfl

def row10247_block_009 : List Nat := []
theorem row10247_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10247 row10247_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10247_block_010 : List Nat := []
theorem row10247_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10247 row10247_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10247_block_011 : List Nat := []
theorem row10247_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10247 row10247_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10247_block_012 : List Nat := []
theorem row10247_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10247 row10247_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10247_block_013 : List Nat := []
theorem row10247_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10247 row10247_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10247_large_left_left : List Nat := row10247_small ++ row10247_block_005
theorem row10247_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10247 row10247_large_left_left :=
  Budget.RowUpperBound.branch row10247_small_covers row10247_block_005_covers

def row10247_large_left_right_right : List Nat := row10247_block_007 ++ row10247_block_008
theorem row10247_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10247 row10247_large_left_right_right :=
  Budget.RowUpperBound.branch row10247_block_007_covers row10247_block_008_covers

def row10247_large_left_right : List Nat := row10247_block_006 ++ row10247_large_left_right_right
theorem row10247_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10247 row10247_large_left_right :=
  Budget.RowUpperBound.branch row10247_block_006_covers row10247_large_left_right_right_covers

def row10247_large_left : List Nat := row10247_large_left_left ++ row10247_large_left_right
theorem row10247_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10247 row10247_large_left :=
  Budget.RowUpperBound.branch row10247_large_left_left_covers row10247_large_left_right_covers

def row10247_large_right_left : List Nat := row10247_block_009 ++ row10247_block_010
theorem row10247_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10247 row10247_large_right_left :=
  Budget.RowUpperBound.branch row10247_block_009_covers row10247_block_010_covers

def row10247_large_right_right_right : List Nat := row10247_block_012 ++ row10247_block_013
theorem row10247_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10247 row10247_large_right_right_right :=
  Budget.RowUpperBound.branch row10247_block_012_covers row10247_block_013_covers

def row10247_large_right_right : List Nat := row10247_block_011 ++ row10247_large_right_right_right
theorem row10247_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10247 row10247_large_right_right :=
  Budget.RowUpperBound.branch row10247_block_011_covers row10247_large_right_right_right_covers

def row10247_large_right : List Nat := row10247_large_right_left ++ row10247_large_right_right
theorem row10247_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10247 row10247_large_right :=
  Budget.RowUpperBound.branch row10247_large_right_left_covers row10247_large_right_right_covers

def row10247_large : List Nat := row10247_large_left ++ row10247_large_right
theorem row10247_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10247 row10247_large :=
  Budget.RowUpperBound.branch row10247_large_left_covers row10247_large_right_covers

def row10247 : List Nat := row10247_large
theorem row10247_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10247 row10247 := row10247_large_covers

def row10253_small : List Nat := []
theorem row10253_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10253 row10253_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10253_block_005 : List Nat := []
def plan10253_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.secondFactor 31), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 31), (.secondFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10253_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10253 row10253_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10253_block_005
  · decide
  · rfl

def row10253_block_006 : List Nat := [10463, 10529, 11069, 12203]
def plan10253_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .possible, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .second, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 19), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, .first, .second, .first, .second, .first, .first, .first, (.secondFactor 37), (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .second, .first, .second, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row10253_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10253 row10253_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10253_block_006
  · decide
  · rfl

def row10253_block_007 : List Nat := [13997, 14177]
def plan10253_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 37), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 17), (.firstFactor 7), (.secondFactor 29), (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.secondFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .second, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, (.firstFactor 19), .possible, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, (.secondFactor 23), .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first]
set_option maxRecDepth 2048 in
theorem row10253_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10253 row10253_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10253_block_007
  · decide
  · rfl

def row10253_block_008 : List Nat := [14831, 15131, 15527]
def plan10253_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, .second, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .second, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 19), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .second, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 31), .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10253_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10253 row10253_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10253_block_008
  · decide
  · rfl

def row10253_block_009 : List Nat := []
theorem row10253_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10253 row10253_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10253_block_010 : List Nat := []
theorem row10253_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10253 row10253_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10253_block_011 : List Nat := []
theorem row10253_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10253 row10253_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10253_block_012 : List Nat := []
theorem row10253_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10253 row10253_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10253_block_013 : List Nat := []
theorem row10253_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10253 row10253_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10253_large_left_left : List Nat := row10253_small ++ row10253_block_005
theorem row10253_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10253 row10253_large_left_left :=
  Budget.RowUpperBound.branch row10253_small_covers row10253_block_005_covers

def row10253_large_left_right_right : List Nat := row10253_block_007 ++ row10253_block_008
theorem row10253_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10253 row10253_large_left_right_right :=
  Budget.RowUpperBound.branch row10253_block_007_covers row10253_block_008_covers

def row10253_large_left_right : List Nat := row10253_block_006 ++ row10253_large_left_right_right
theorem row10253_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10253 row10253_large_left_right :=
  Budget.RowUpperBound.branch row10253_block_006_covers row10253_large_left_right_right_covers

def row10253_large_left : List Nat := row10253_large_left_left ++ row10253_large_left_right
theorem row10253_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10253 row10253_large_left :=
  Budget.RowUpperBound.branch row10253_large_left_left_covers row10253_large_left_right_covers

def row10253_large_right_left : List Nat := row10253_block_009 ++ row10253_block_010
theorem row10253_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10253 row10253_large_right_left :=
  Budget.RowUpperBound.branch row10253_block_009_covers row10253_block_010_covers

def row10253_large_right_right_right : List Nat := row10253_block_012 ++ row10253_block_013
theorem row10253_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10253 row10253_large_right_right_right :=
  Budget.RowUpperBound.branch row10253_block_012_covers row10253_block_013_covers

def row10253_large_right_right : List Nat := row10253_block_011 ++ row10253_large_right_right_right
theorem row10253_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10253 row10253_large_right_right :=
  Budget.RowUpperBound.branch row10253_block_011_covers row10253_large_right_right_right_covers

def row10253_large_right : List Nat := row10253_large_right_left ++ row10253_large_right_right
theorem row10253_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10253 row10253_large_right :=
  Budget.RowUpperBound.branch row10253_large_right_left_covers row10253_large_right_right_covers

def row10253_large : List Nat := row10253_large_left ++ row10253_large_right
theorem row10253_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10253 row10253_large :=
  Budget.RowUpperBound.branch row10253_large_left_covers row10253_large_right_covers

def row10253 : List Nat := row10253_large
theorem row10253_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10253 row10253 := row10253_large_covers

def row10259_small : List Nat := []
theorem row10259_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10259 row10259_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10259_block_005 : List Nat := []
def plan10259_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 29), .first, .first]
set_option maxRecDepth 2048 in
theorem row10259_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10259 row10259_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10259_block_005
  · decide
  · rfl

def row10259_block_006 : List Nat := [10799, 12101]
def plan10259_block_006 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 37), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 37), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .second, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .second, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row10259_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10259 row10259_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10259_block_006
  · decide
  · rfl

def row10259_block_007 : List Nat := [12893, 12983, 13121, 14159, 14321]
def plan10259_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, .possible, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .possible, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row10259_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10259 row10259_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10259_block_007
  · decide
  · rfl

def row10259_block_008 : List Nat := [15083, 15131, 15383]
def plan10259_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 13), .first, .second, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.firstFactor 31), (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10259_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10259 row10259_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10259_block_008
  · decide
  · rfl

def row10259_block_009 : List Nat := []
theorem row10259_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10259 row10259_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10259_block_010 : List Nat := []
theorem row10259_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10259 row10259_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10259_block_011 : List Nat := []
theorem row10259_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10259 row10259_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10259_block_012 : List Nat := []
theorem row10259_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10259 row10259_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10259_block_013 : List Nat := []
theorem row10259_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10259 row10259_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10259_large_left_left : List Nat := row10259_small ++ row10259_block_005
theorem row10259_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10259 row10259_large_left_left :=
  Budget.RowUpperBound.branch row10259_small_covers row10259_block_005_covers

def row10259_large_left_right_right : List Nat := row10259_block_007 ++ row10259_block_008
theorem row10259_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10259 row10259_large_left_right_right :=
  Budget.RowUpperBound.branch row10259_block_007_covers row10259_block_008_covers

def row10259_large_left_right : List Nat := row10259_block_006 ++ row10259_large_left_right_right
theorem row10259_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10259 row10259_large_left_right :=
  Budget.RowUpperBound.branch row10259_block_006_covers row10259_large_left_right_right_covers

def row10259_large_left : List Nat := row10259_large_left_left ++ row10259_large_left_right
theorem row10259_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10259 row10259_large_left :=
  Budget.RowUpperBound.branch row10259_large_left_left_covers row10259_large_left_right_covers

def row10259_large_right_left : List Nat := row10259_block_009 ++ row10259_block_010
theorem row10259_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10259 row10259_large_right_left :=
  Budget.RowUpperBound.branch row10259_block_009_covers row10259_block_010_covers

def row10259_large_right_right_right : List Nat := row10259_block_012 ++ row10259_block_013
theorem row10259_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10259 row10259_large_right_right_right :=
  Budget.RowUpperBound.branch row10259_block_012_covers row10259_block_013_covers

def row10259_large_right_right : List Nat := row10259_block_011 ++ row10259_large_right_right_right
theorem row10259_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10259 row10259_large_right_right :=
  Budget.RowUpperBound.branch row10259_block_011_covers row10259_large_right_right_right_covers

def row10259_large_right : List Nat := row10259_large_right_left ++ row10259_large_right_right
theorem row10259_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10259 row10259_large_right :=
  Budget.RowUpperBound.branch row10259_large_right_left_covers row10259_large_right_right_covers

def row10259_large : List Nat := row10259_large_left ++ row10259_large_right
theorem row10259_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10259 row10259_large :=
  Budget.RowUpperBound.branch row10259_large_left_covers row10259_large_right_covers

def row10259 : List Nat := row10259_large
theorem row10259_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10259 row10259 := row10259_large_covers

end ProjectEulerSolutions.PrimePairLarge
