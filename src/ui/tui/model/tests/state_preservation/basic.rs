use crate::ui::tui::model::Model;

/// Test quit functionality
#[test]
fn test_quit_functionality() {
    let mut model = Model::new();

    assert!(!model.should_quit);

    model.quit();

    assert!(model.should_quit);
}
