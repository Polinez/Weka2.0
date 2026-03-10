from django.contrib import admin
from .models import PreprocessingType, PreprocessingPipeline, PreprocessingStep

@admin.register(PreprocessingType)
class PreprocessingTypeAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "description")
    search_fields = ("name",)


class PreprocessingStepInline(admin.TabularInline):
    model = PreprocessingStep
    extra = 0
    readonly_fields = ("applied_at",)

@admin.register(PreprocessingPipeline)
class PreprocessingPipelineAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "dataset",
        "is_active",
        "created_at",
        "processed_file_path",
        "processed_train_path",
        "processed_test_path"
    )
    list_filter = ("is_active",)
    readonly_fields = ("created_at", "output_columns_metadata", "split_config")
    inlines = [PreprocessingStepInline]