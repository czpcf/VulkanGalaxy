#include <vulkan/vulkan.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <iostream>
#include <vector>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <optional>
#include "utils.hpp"

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;
const std::string APP_NAME = "VulkanGalaxy";
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
	std::optional<uint32_t> uniformFamily; // uniform = graphics + compute
	std::optional<uint32_t> presentFamily;

	bool isComplete() const {
		return uniformFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapchainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

static std::vector<const char*> getRequiredExtensions();
static bool isDeviceSuitable(VkPhysicalDevice, VkSurfaceKHR&);
static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice, VkSurfaceKHR&);
static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice, VkSurfaceKHR&);
static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&);
static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>&);
static VkExtent2D chooseSwapExtent(GLFWwindow*, const VkSurfaceCapabilitiesKHR&);
static VkImageView createImageView(VkDevice, VkImage, VkFormat, VkImageAspectFlags, uint32_t);
static uint32_t findMemoryType(VkPhysicalDevice, uint32_t, VkMemoryPropertyFlags);
static VkFormat findSupportedFormat(VkPhysicalDevice, const std::vector<VkFormat>&, VkImageTiling, VkFormatFeatureFlags);
static bool hasStencilComponent(VkFormat);
static VkFormat findDepthFormat(VkPhysicalDevice);


class Image2D {
public:
	// create a 2D image and bind it to the memory
	void create(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
		VkSampleCountFlagBits numSamples, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags) {
		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = format,
			.extent = {
				.width = width,
				.height = height,
				.depth = 1,
			},
			.mipLevels = mipLevels,
			.arrayLayers = 1,
			.samples = numSamples,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		UT_CHECK_ERR(vkCreateImage(device, &imageInfo, nullptr, &image) == VK_SUCCESS, "failed to create image");
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);
		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties),
		};
		UT_CHECK_ERR(vkAllocateMemory(device, &allocInfo, nullptr, &memory) == VK_SUCCESS, "failed to allocate image momory");
		UT_CHECK_ERR(vkBindImageMemory(device, image, memory, 0) == VK_SUCCESS, "failed to bind image to memory");
		imageView = createImageView(device, image, format, aspectFlags, mipLevels);
	}
	void destroy(VkDevice device) const {
		vkDestroyImageView(device, imageView, nullptr);
		vkDestroyImage(device, image, nullptr);
		vkFreeMemory(device, memory, nullptr);
	}
private:
	VkImage image;
	VkImageView imageView;
	VkDeviceMemory memory;
};

class App {
public:
	App() {}
	void run() {
		initWindow();
		initVulkan();
		std::cerr << "app intialized" << std::endl;
		mainLoop();
		cleanup();
	}
private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkDevice device;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	QueueFamilyIndices indices;
	SwapchainSupportDetails details;
	VkQueue uniformQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	VkExtent2D swapchainExtent;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	Image2D colorImage;
	Image2D depthImage;

	// initialize the window using glfw
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME.c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// TODO
	// recreate swapchain upon resizing the window
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	}

	// initialize vulkan
	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapchain();
		createImageViews();

		createCommandPool();
		createCommandBuffers();
		createResources();
	}

	// create vulkan instance for all
	void createInstance() {
		VkApplicationInfo appInfo{
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = APP_NAME.c_str(),
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0,
		};

		if (enableValidationLayers) {
			UT_CHECK_ERR(checkRequiredLayersSupport(validationLayers), "do not support validation layers");
		}
		auto extensions = getRequiredExtensions();
		VkInstanceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data(),
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);
			createInfo.pNext = (VkDebugUtilsMessengerEXT*)&createInfo;
		} else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		UT_CHECK_ERR(vkCreateInstance(&createInfo, nullptr, &instance) == VK_SUCCESS, "failed to create instace");
	}

	// setup the debug messenger, the debug extension must be enabled
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		UT_CHECK_ERR(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) == VK_SUCCESS, "failed to setup debug messenger, check whether debug ext is not supported");
	}

	// create surface and bind it to the instance
	void createSurface() {
		UT_CHECK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface) == VK_SUCCESS, "failed to create window surface");
	}

	// choose a suitable device
	void pickPhysicalDevice() {
		uint32_t cnt = 0;
		vkEnumeratePhysicalDevices(instance, &cnt, nullptr);
		UT_CHECK_ERR(cnt != 0, "failed to find GPUs with Vulkan support");
		
		std::vector<VkPhysicalDevice> devices(cnt);
		vkEnumeratePhysicalDevices(instance, &cnt, devices.data());
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			indices = findQueueFamilyProperties(device, surface);
			details = querySwapchainSupport(device, surface);
			if (isDeviceSuitable(device, surface)) {
				physicalDevice = device;
				// TODO: add msaa
				break;
			}
		}
		UT_CHECK_ERR(physicalDevice != VK_NULL_HANDLE, "failed to find a suitabale GPU");
	}

	// create logical device with queues and enable device features
	void createLogicalDevice() {
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.uniformFamily.value(), indices.presentFamily.value() };
		float queuePriority = 1.0f;

		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamily,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		// TODO: add msaa feature

		VkDeviceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
			.pQueueCreateInfos = queueCreateInfos.data(),
			.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
			.ppEnabledExtensionNames = deviceExtensions.data(),
			.pEnabledFeatures = nullptr,
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}
		UT_CHECK_ERR(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) == VK_SUCCESS, "failed to create logical device");

		// fetch queues
		vkGetDeviceQueue(device, indices.uniformFamily.value(), 0, &uniformQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// swapchain
	void createSwapchain() {
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(details.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(details.presentModes);
		VkExtent2D extent = chooseSwapExtent(window, details.capabilities);
		uint32_t imageCount = details.capabilities.minImageCount + 1;
		if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount) {
			imageCount = details.capabilities.maxImageCount;
		}
		VkSwapchainCreateInfoKHR createInfo{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.preTransform = details.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = presentMode,
			.clipped = VK_TRUE,
			.oldSwapchain = VK_NULL_HANDLE,
		};

		uint32_t queueFamilyIndices[] = { indices.uniformFamily.value(), indices.presentFamily.value() };
		if (indices.uniformFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}
		UT_CHECK_ERR(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) == VK_SUCCESS, "failed to create swapchain");
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
		swapchainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
		swapchainImageFormat = surfaceFormat.format;
		swapchainExtent = extent;
	}

	// image views to swapchain
	void createImageViews() {
		swapchainImageViews.resize(swapchainImages.size());
		for (int i = 0; i < swapchainImages.size(); ++i) {
			swapchainImageViews[i] = createImageView(device, swapchainImages[i], swapchainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	// command pool
	void createCommandPool() {
		VkCommandPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = indices.uniformFamily.value(),
		};
		UT_CHECK_ERR(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) == VK_SUCCESS, "failed to create command pool");
	}

	// command buffer
	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (uint32_t)commandBuffers.size(),
		};
		UT_CHECK_ERR(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) == VK_SUCCESS, "failed to allocate command buffers");
	}

	// color
	void createResources() {
		VkFormat colorFormat = swapchainImageFormat;
		uint32_t width = swapchainExtent.width;
		uint32_t height = swapchainExtent.height;
		// TODO: msaa
		VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
		colorImage.create(physicalDevice, device, width, height, 1, colorFormat,
			samples,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT);
		auto depthFormat = findDepthFormat(physicalDevice);
		depthImage.create(physicalDevice, device, width, height, 1, depthFormat,
			samples,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT);
	}

	void cleanupSwapchain() {
		colorImage.destroy(device);
		depthImage.destroy(device);
		for (auto imageView : swapchainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapchain, nullptr);
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		cleanupSwapchain();
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		}
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyDevice(device, nullptr);
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		exit(0);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
};


// get glfw & validation extensions
static std::vector<const char*> getRequiredExtensions() {
	uint32_t cnt = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&cnt);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + cnt);
	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}

// chech if the device supports swapchain, extensions and features
static bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices = findQueueFamilyProperties(device, surface);
	bool extensionsSupported = checkRequiredExtensionsSupport(device, deviceExtensions);
	bool swapChainAdequate = false;
	if (extensionsSupported) { // swapchain is included
		SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, surface);
		swapChainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
	}
	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
	// TODO: check samplerAnisotropy if msaa is required
	return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	SwapchainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
	uint32_t cnt;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.formats.resize(cnt);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, details.formats.data());
	}
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.presentModes.resize(cnt);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, details.presentModes.data());
	}
	return details;
}

static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices;
	uint32_t cnt;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(cnt);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, queueFamilies.data());
	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			indices.uniformFamily = i;
		}
		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (presentSupport) {
			indices.presentFamily = i;
		}
		++i;
	}
	return indices;
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
	for (const auto& format : availableFormats) {
		if (format.format == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return format;
		}
	}
	return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
	for (const auto& mode : availablePresentModes) {
		if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return mode;
		}
	}
	return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	}
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	VkExtent2D actualExtent = {
		static_cast<uint32_t>(width),
		static_cast<uint32_t>(height),
	};
	actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
	actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	return actualExtent;
}

// create a 2D image view
static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLeveles) {
	VkImageViewCreateInfo viewInfo{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = aspectFlags,
			.baseMipLevel = 0,
			.levelCount = mipLeveles,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkImageView imageView;
	UT_CHECK_ERR(vkCreateImageView(device, &viewInfo, nullptr, &imageView) == VK_SUCCESS, "failed to create texture image view");
	return imageView;
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}
	throw std::runtime_error("failed to find suitable memory type");
}

static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
	for (VkFormat format : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		} else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		} else {
			throw std::runtime_error("not supported tiling format");
		}
	}
	throw std::runtime_error("failed to find supported format");
}

static bool hasStencilComponent(VkFormat format) {
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

static VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
	return findSupportedFormat(physicalDevice,
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

int main() {
	App app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
