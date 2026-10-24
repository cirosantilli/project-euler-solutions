import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9949_small : List Nat := []
theorem row9949_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9949 row9949_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9949_block_005 : List Nat := [10009]
def plan9949_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), (.secondFactor 19), .first, .possible, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 17)]
set_option maxRecDepth 2048 in
theorem row9949_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9949 row9949_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9949_block_005
  · decide
  · rfl

def row9949_block_006 : List Nat := [10639, 11257]
def plan9949_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 29), .first, .possible, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 19), (.firstFactor 11), .first, .second, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row9949_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9949 row9949_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9949_block_006
  · decide
  · rfl

def row9949_block_007 : List Nat := [12619, 12703, 13687]
def plan9949_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.secondFactor 13), .possible, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 29), (.firstFactor 19), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 29), .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), (.firstFactor 19)]
set_option maxRecDepth 2048 in
theorem row9949_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9949 row9949_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9949_block_007
  · decide
  · rfl

def row9949_block_008 : List Nat := [14593, 15091, 15139]
def plan9949_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .possible, .first, .first, (.secondFactor 29), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .second, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9949_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9949 row9949_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9949_block_008
  · decide
  · rfl

def row9949_block_009 : List Nat := []
theorem row9949_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9949 row9949_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9949_block_010 : List Nat := []
theorem row9949_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9949 row9949_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9949_block_011 : List Nat := []
theorem row9949_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9949 row9949_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9949_block_012 : List Nat := []
theorem row9949_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9949 row9949_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9949_block_013 : List Nat := []
theorem row9949_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9949 row9949_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9949_large_left_left : List Nat := row9949_small ++ row9949_block_005
theorem row9949_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9949 row9949_large_left_left :=
  Budget.RowUpperBound.branch row9949_small_covers row9949_block_005_covers

def row9949_large_left_right_right : List Nat := row9949_block_007 ++ row9949_block_008
theorem row9949_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9949 row9949_large_left_right_right :=
  Budget.RowUpperBound.branch row9949_block_007_covers row9949_block_008_covers

def row9949_large_left_right : List Nat := row9949_block_006 ++ row9949_large_left_right_right
theorem row9949_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9949 row9949_large_left_right :=
  Budget.RowUpperBound.branch row9949_block_006_covers row9949_large_left_right_right_covers

def row9949_large_left : List Nat := row9949_large_left_left ++ row9949_large_left_right
theorem row9949_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9949 row9949_large_left :=
  Budget.RowUpperBound.branch row9949_large_left_left_covers row9949_large_left_right_covers

def row9949_large_right_left : List Nat := row9949_block_009 ++ row9949_block_010
theorem row9949_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9949 row9949_large_right_left :=
  Budget.RowUpperBound.branch row9949_block_009_covers row9949_block_010_covers

def row9949_large_right_right_right : List Nat := row9949_block_012 ++ row9949_block_013
theorem row9949_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9949 row9949_large_right_right_right :=
  Budget.RowUpperBound.branch row9949_block_012_covers row9949_block_013_covers

def row9949_large_right_right : List Nat := row9949_block_011 ++ row9949_large_right_right_right
theorem row9949_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9949 row9949_large_right_right :=
  Budget.RowUpperBound.branch row9949_block_011_covers row9949_large_right_right_right_covers

def row9949_large_right : List Nat := row9949_large_right_left ++ row9949_large_right_right
theorem row9949_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9949 row9949_large_right :=
  Budget.RowUpperBound.branch row9949_large_right_left_covers row9949_large_right_right_covers

def row9949_large : List Nat := row9949_large_left ++ row9949_large_right
theorem row9949_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9949 row9949_large :=
  Budget.RowUpperBound.branch row9949_large_left_covers row9949_large_right_covers

def row9949 : List Nat := row9949_large
theorem row9949_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9949 row9949 := row9949_large_covers

def row9967_small : List Nat := []
theorem row9967_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9967 row9967_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9967_block_005 : List Nat := []
def plan9967_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .second, .first, .second, .first, .first, (.firstFactor 17), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9967_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9967 row9967_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9967_block_005
  · decide
  · rfl

def row9967_block_006 : List Nat := [10513, 10837, 12109, 12391]
def plan9967_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .possible, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 29), .possible, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .second, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 29), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9967_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9967 row9967_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9967_block_006
  · decide
  · rfl

def row9967_block_007 : List Nat := [13327, 13513, 13693, 13873, 14281]
def plan9967_block_007 : List Budget.Choice := [(.secondFactor 19), .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 31), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .second, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 29), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 37), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 31), .second]
set_option maxRecDepth 2048 in
theorem row9967_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9967 row9967_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9967_block_007
  · decide
  · rfl

def row9967_block_008 : List Nat := [15607, 15667, 16057]
def plan9967_block_008 : List Budget.Choice := [(.secondFactor 13), .first, .second, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.secondFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 31), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .possible, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), .possible, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9967_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9967 row9967_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9967_block_008
  · decide
  · rfl

def row9967_block_009 : List Nat := []
theorem row9967_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9967 row9967_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9967_block_010 : List Nat := []
theorem row9967_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9967 row9967_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9967_block_011 : List Nat := []
theorem row9967_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9967 row9967_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9967_block_012 : List Nat := []
theorem row9967_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9967 row9967_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9967_block_013 : List Nat := []
theorem row9967_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9967 row9967_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9967_large_left_left : List Nat := row9967_small ++ row9967_block_005
theorem row9967_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9967 row9967_large_left_left :=
  Budget.RowUpperBound.branch row9967_small_covers row9967_block_005_covers

def row9967_large_left_right_right : List Nat := row9967_block_007 ++ row9967_block_008
theorem row9967_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9967 row9967_large_left_right_right :=
  Budget.RowUpperBound.branch row9967_block_007_covers row9967_block_008_covers

def row9967_large_left_right : List Nat := row9967_block_006 ++ row9967_large_left_right_right
theorem row9967_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9967 row9967_large_left_right :=
  Budget.RowUpperBound.branch row9967_block_006_covers row9967_large_left_right_right_covers

def row9967_large_left : List Nat := row9967_large_left_left ++ row9967_large_left_right
theorem row9967_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9967 row9967_large_left :=
  Budget.RowUpperBound.branch row9967_large_left_left_covers row9967_large_left_right_covers

def row9967_large_right_left : List Nat := row9967_block_009 ++ row9967_block_010
theorem row9967_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9967 row9967_large_right_left :=
  Budget.RowUpperBound.branch row9967_block_009_covers row9967_block_010_covers

def row9967_large_right_right_right : List Nat := row9967_block_012 ++ row9967_block_013
theorem row9967_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9967 row9967_large_right_right_right :=
  Budget.RowUpperBound.branch row9967_block_012_covers row9967_block_013_covers

def row9967_large_right_right : List Nat := row9967_block_011 ++ row9967_large_right_right_right
theorem row9967_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9967 row9967_large_right_right :=
  Budget.RowUpperBound.branch row9967_block_011_covers row9967_large_right_right_right_covers

def row9967_large_right : List Nat := row9967_large_right_left ++ row9967_large_right_right
theorem row9967_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9967 row9967_large_right :=
  Budget.RowUpperBound.branch row9967_large_right_left_covers row9967_large_right_right_covers

def row9967_large : List Nat := row9967_large_left ++ row9967_large_right
theorem row9967_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9967 row9967_large :=
  Budget.RowUpperBound.branch row9967_large_left_covers row9967_large_right_covers

def row9967 : List Nat := row9967_large
theorem row9967_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9967 row9967 := row9967_large_covers

def row9973_small : List Nat := []
theorem row9973_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9973 row9973_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9973_block_005 : List Nat := []
def plan9973_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9973_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9973 row9973_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9973_block_005
  · decide
  · rfl

def row9973_block_006 : List Nat := [10459, 11677, 12343]
def plan9973_block_006 : List Budget.Choice := [.first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .possible, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 31), .first, (.firstFactor 17), .first, .first, (.secondFactor 23), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 17), .first, .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.secondFactor 23), .first]
set_option maxRecDepth 2048 in
theorem row9973_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9973 row9973_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9973_block_006
  · decide
  · rfl

def row9973_block_007 : List Nat := [13063, 13417, 14107, 14449]
def plan9973_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .second, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .second, (.secondFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 29), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 23), .first, .first, .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .second, .first, (.secondFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .possible, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9973_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9973 row9973_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9973_block_007
  · decide
  · rfl

def row9973_block_008 : List Nat := [14827, 15373, 15391]
def plan9973_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .possible, .first, .first, (.firstFactor 13), .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .possible, .first, .first, .possible, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .second, (.secondFactor 17), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9973_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9973 row9973_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9973_block_008
  · decide
  · rfl

def row9973_block_009 : List Nat := []
theorem row9973_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9973 row9973_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row9973_block_010 : List Nat := []
theorem row9973_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9973 row9973_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9973_block_011 : List Nat := []
theorem row9973_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9973 row9973_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9973_block_012 : List Nat := []
theorem row9973_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9973 row9973_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9973_block_013 : List Nat := []
theorem row9973_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9973 row9973_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9973_large_left_left : List Nat := row9973_small ++ row9973_block_005
theorem row9973_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9973 row9973_large_left_left :=
  Budget.RowUpperBound.branch row9973_small_covers row9973_block_005_covers

def row9973_large_left_right_right : List Nat := row9973_block_007 ++ row9973_block_008
theorem row9973_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9973 row9973_large_left_right_right :=
  Budget.RowUpperBound.branch row9973_block_007_covers row9973_block_008_covers

def row9973_large_left_right : List Nat := row9973_block_006 ++ row9973_large_left_right_right
theorem row9973_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9973 row9973_large_left_right :=
  Budget.RowUpperBound.branch row9973_block_006_covers row9973_large_left_right_right_covers

def row9973_large_left : List Nat := row9973_large_left_left ++ row9973_large_left_right
theorem row9973_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9973 row9973_large_left :=
  Budget.RowUpperBound.branch row9973_large_left_left_covers row9973_large_left_right_covers

def row9973_large_right_left : List Nat := row9973_block_009 ++ row9973_block_010
theorem row9973_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9973 row9973_large_right_left :=
  Budget.RowUpperBound.branch row9973_block_009_covers row9973_block_010_covers

def row9973_large_right_right_right : List Nat := row9973_block_012 ++ row9973_block_013
theorem row9973_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9973 row9973_large_right_right_right :=
  Budget.RowUpperBound.branch row9973_block_012_covers row9973_block_013_covers

def row9973_large_right_right : List Nat := row9973_block_011 ++ row9973_large_right_right_right
theorem row9973_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9973 row9973_large_right_right :=
  Budget.RowUpperBound.branch row9973_block_011_covers row9973_large_right_right_right_covers

def row9973_large_right : List Nat := row9973_large_right_left ++ row9973_large_right_right
theorem row9973_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9973 row9973_large_right :=
  Budget.RowUpperBound.branch row9973_large_right_left_covers row9973_large_right_right_covers

def row9973_large : List Nat := row9973_large_left ++ row9973_large_right
theorem row9973_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9973 row9973_large :=
  Budget.RowUpperBound.branch row9973_large_left_covers row9973_large_right_covers

def row9973 : List Nat := row9973_large
theorem row9973_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9973 row9973 := row9973_large_covers

def row10007_small : List Nat := []
theorem row10007_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10007 row10007_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10007_block_005 : List Nat := []
def plan10007_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 37), .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row10007_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10007 row10007_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan10007_block_005
  · decide
  · rfl

def row10007_block_006 : List Nat := [10709, 12281, 12347]
def plan10007_block_006 : List Budget.Choice := [(.secondFactor 13), .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.secondFactor 29), (.firstFactor 29), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 23), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .second, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 37), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 19), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 29), (.firstFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .possible, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row10007_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10007 row10007_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10007_block_006
  · decide
  · rfl

def row10007_block_007 : List Nat := [14207]
def plan10007_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 29), .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10007_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10007 row10007_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10007_block_007
  · decide
  · rfl

def row10007_block_008 : List Nat := [15473]
def plan10007_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 37), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), (.firstFactor 19), .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 19), .first, .second, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .second, .first, .first, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10007_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10007 row10007_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10007_block_008
  · decide
  · rfl

def row10007_block_009 : List Nat := []
theorem row10007_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10007 row10007_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10007_block_010 : List Nat := []
theorem row10007_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10007 row10007_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10007_block_011 : List Nat := []
theorem row10007_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10007 row10007_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10007_block_012 : List Nat := []
theorem row10007_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10007 row10007_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10007_block_013 : List Nat := []
theorem row10007_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10007 row10007_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10007_large_left_left : List Nat := row10007_small ++ row10007_block_005
theorem row10007_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10007 row10007_large_left_left :=
  Budget.RowUpperBound.branch row10007_small_covers row10007_block_005_covers

def row10007_large_left_right_right : List Nat := row10007_block_007 ++ row10007_block_008
theorem row10007_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10007 row10007_large_left_right_right :=
  Budget.RowUpperBound.branch row10007_block_007_covers row10007_block_008_covers

def row10007_large_left_right : List Nat := row10007_block_006 ++ row10007_large_left_right_right
theorem row10007_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10007 row10007_large_left_right :=
  Budget.RowUpperBound.branch row10007_block_006_covers row10007_large_left_right_right_covers

def row10007_large_left : List Nat := row10007_large_left_left ++ row10007_large_left_right
theorem row10007_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10007 row10007_large_left :=
  Budget.RowUpperBound.branch row10007_large_left_left_covers row10007_large_left_right_covers

def row10007_large_right_left : List Nat := row10007_block_009 ++ row10007_block_010
theorem row10007_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10007 row10007_large_right_left :=
  Budget.RowUpperBound.branch row10007_block_009_covers row10007_block_010_covers

def row10007_large_right_right_right : List Nat := row10007_block_012 ++ row10007_block_013
theorem row10007_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10007 row10007_large_right_right_right :=
  Budget.RowUpperBound.branch row10007_block_012_covers row10007_block_013_covers

def row10007_large_right_right : List Nat := row10007_block_011 ++ row10007_large_right_right_right
theorem row10007_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10007 row10007_large_right_right :=
  Budget.RowUpperBound.branch row10007_block_011_covers row10007_large_right_right_right_covers

def row10007_large_right : List Nat := row10007_large_right_left ++ row10007_large_right_right
theorem row10007_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10007 row10007_large_right :=
  Budget.RowUpperBound.branch row10007_large_right_left_covers row10007_large_right_right_covers

def row10007_large : List Nat := row10007_large_left ++ row10007_large_right
theorem row10007_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10007 row10007_large :=
  Budget.RowUpperBound.branch row10007_large_left_covers row10007_large_right_covers

def row10007 : List Nat := row10007_large
theorem row10007_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10007 row10007 := row10007_large_covers

end ProjectEulerSolutions.PrimePairLarge
